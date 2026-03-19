"""Unified product classifier with auto-detection of best available model.

Priority order:
  1. DINOv2 supervised head (direct 356-class prediction)
  2. DINOv2 embedding matching (cosine similarity)
  3. EfficientNet-B3 embedding matching (cosine similarity)
  4. No classifier (detection-only, category_id=0)

When both supervised and embedding models are available, runs dual-mode
classification with weighted combination (0.6 supervised + 0.4 embedding).

Uses pathlib only (no `import os`). Compatible with timm==0.9.12.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

TEMPERATURE = 0.07
NUM_CLASSES = 356


class ProductClassifier:
    """Auto-loads the best available classifier from model_dir."""

    def __init__(self, model_dir: Path, device: str):
        self.device = device
        self._mode = "none"

        # Try loading models in priority order
        self._supervised_model = None
        self._supervised_head = None
        self._embedding_model = None
        self._ref_embeddings = None
        self._valid_mask = None
        self._supervised_transform = None
        self._embedding_transform = None

        # Try consolidated file first (dinov2_all.pt = classifier + embeddings in one file)
        self._try_load_consolidated(model_dir)

        # Fallback: individual files
        if self._supervised_model is None:
            self._try_load_dinov2_supervised(model_dir)

        if self._embedding_model is None:
            self._try_load_dinov2_embedding(model_dir)

        # Priority 3: EfficientNet-B3 embedding matching
        if self._embedding_model is None:
            self._try_load_efficientnet_embedding(model_dir)

        # Determine mode
        if self._supervised_model is not None and self._embedding_model is not None:
            self._mode = "dual"
            print("[CLASSIFIER] Mode: dual (dinov2_supervised + embedding)")
        elif self._supervised_model is not None:
            self._mode = "dinov2_supervised"
            print("[CLASSIFIER] Mode: dinov2_supervised")
        elif self._embedding_model is not None:
            # Check which embedding model was loaded
            if self._embedding_is_dinov2:
                self._mode = "dinov2_embedding"
                print("[CLASSIFIER] Mode: dinov2_embedding")
            else:
                self._mode = "efficientnet_embedding"
                print("[CLASSIFIER] Mode: efficientnet_embedding")
        else:
            self._mode = "none"
            print("[CLASSIFIER] Mode: none (detection-only, category_id=0)")

    def _try_load_consolidated(self, model_dir: Path) -> None:
        """Load from consolidated dinov2_all.pt (classifier + embeddings in one file)."""
        all_path = model_dir / "dinov2_all.pt"
        if not all_path.exists():
            return

        try:
            import timm
        except ImportError:
            return

        try:
            data = torch.load(str(all_path), map_location=self.device, weights_only=False)

            # Load supervised classifier
            if "classifier" in data:
                cls_data = data["classifier"]
                model = timm.create_model(
                    "vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0, img_size=224,
                )
                head = nn.Linear(cls_data.get("embed_dim", 768), NUM_CLASSES)

                if "backbone" in cls_data and "classifier_head" in cls_data:
                    model.load_state_dict(cls_data["backbone"], strict=False)
                    head.load_state_dict(cls_data["classifier_head"])
                elif "backbone" in cls_data and "head" in cls_data:
                    model.load_state_dict(cls_data["backbone"], strict=False)
                    head.load_state_dict(cls_data["head"])

                model = model.to(self.device).eval()
                head = head.to(self.device).eval()
                self._supervised_model = model
                self._supervised_head = head
                self._supervised_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                print(f"[CLASSIFIER] Loaded DINOv2 supervised from consolidated file")

            # Load embedding model
            if "embedding_backbone" in data and "product_embeddings" in data:
                emb_model = timm.create_model(
                    "vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0, img_size=224,
                )
                emb_model.load_state_dict(data["embedding_backbone"], strict=False)
                emb_model = emb_model.to(self.device).eval()

                ref_embeddings = data["product_embeddings"].to(self.device)
                valid_mask = ref_embeddings.norm(dim=1) > 0.1

                self._embedding_model = emb_model
                self._ref_embeddings = ref_embeddings
                self._valid_mask = valid_mask
                self._embedding_is_dinov2 = True
                self._embedding_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                print(f"[CLASSIFIER] Loaded DINOv2 embeddings from consolidated file")

        except Exception as e:
            print(f"[CLASSIFIER] Failed to load consolidated: {e}")

    def _try_load_dinov2_supervised(self, model_dir: Path) -> None:
        """Load DINOv2 backbone + linear head for direct classification."""
        weights_path = model_dir / "dinov2_classifier_weights.pt"
        if not weights_path.exists():
            return

        try:
            import timm
        except ImportError:
            print("[CLASSIFIER] timm not available — skipping DINOv2 supervised")
            return

        try:
            model = timm.create_model(
                "vit_base_patch14_dinov2.lvd142m",
                pretrained=False,
                num_classes=0,
                img_size=224,
            )
            model = model.to(self.device).eval()

            head = nn.Linear(768, NUM_CLASSES)

            state_dict = torch.load(
                str(weights_path), map_location=self.device, weights_only=True
            )

            # Support both combined and separate state dicts
            if "backbone" in state_dict and "head" in state_dict:
                model.load_state_dict(state_dict["backbone"], strict=False)
                head.load_state_dict(state_dict["head"])
            elif any(k.startswith("head.") for k in state_dict):
                # Combined state dict with head.weight, head.bias + backbone keys
                backbone_sd = {
                    k: v for k, v in state_dict.items() if not k.startswith("head.")
                }
                head_sd = {
                    k.replace("head.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("head.")
                }
                if backbone_sd:
                    model.load_state_dict(backbone_sd, strict=False)
                head.load_state_dict(head_sd)
            else:
                # Assume it's a full model state dict; try loading directly
                model.load_state_dict(state_dict, strict=False)

            head = head.to(self.device).eval()

            self._supervised_model = model
            self._supervised_head = head
            self._supervised_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])
            print(f"[CLASSIFIER] Loaded DINOv2 supervised head from {weights_path.name}")
        except Exception as e:
            print(f"[CLASSIFIER] Failed to load DINOv2 supervised: {e}")

    def _try_load_dinov2_embedding(self, model_dir: Path) -> None:
        """Load DINOv2 backbone for embedding matching."""
        weights_path = model_dir / "dinov2_embeddings_weights.pt"
        embeddings_path = model_dir / "dinov2_product_embeddings.npy"

        if not weights_path.exists() or not embeddings_path.exists():
            return

        try:
            import timm
        except ImportError:
            print("[CLASSIFIER] timm not available — skipping DINOv2 embedding")
            return

        try:
            model = timm.create_model(
                "vit_base_patch14_dinov2.lvd142m",
                pretrained=False,
                num_classes=0,
                img_size=224,
            )
            state_dict = torch.load(
                str(weights_path), map_location=self.device, weights_only=True
            )
            model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device).eval()

            embeddings = np.load(str(embeddings_path))
            ref_embeddings = torch.from_numpy(embeddings).to(self.device)
            valid_mask = ref_embeddings.norm(dim=1) > 0.1

            self._embedding_model = model
            self._ref_embeddings = ref_embeddings
            self._valid_mask = valid_mask
            self._embedding_is_dinov2 = True
            self._embedding_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])
            print(f"[CLASSIFIER] Loaded DINOv2 embedding model from {weights_path.name}")
            print(f"[CLASSIFIER] Loaded {ref_embeddings.shape[0]} DINOv2 reference embeddings")
        except Exception as e:
            print(f"[CLASSIFIER] Failed to load DINOv2 embedding: {e}")

    def _try_load_efficientnet_embedding(self, model_dir: Path) -> None:
        """Load EfficientNet-B3 for embedding matching."""
        config_path = model_dir / "embedding_config.json"
        embeddings_path = model_dir / "product_embeddings.npy"

        if not config_path.exists() or not embeddings_path.exists():
            return

        try:
            import timm
        except ImportError:
            print("[CLASSIFIER] timm not available — skipping EfficientNet")
            return

        try:
            with open(str(config_path)) as f:
                config = json.load(f)

            weights_path = model_dir / "efficientnet_b3_weights.pt"
            model = timm.create_model(config["model_name"], pretrained=False, num_classes=0)

            if weights_path.exists():
                state_dict = torch.load(
                    str(weights_path), map_location=self.device, weights_only=True
                )
                model.load_state_dict(state_dict, strict=False)
                print(f"[CLASSIFIER] Loaded EfficientNet-B3 weights from {weights_path.name}")
            else:
                print("[CLASSIFIER] WARNING: No EfficientNet weights — classification will be poor")

            model = model.to(self.device).eval()

            embeddings = np.load(str(embeddings_path))
            ref_embeddings = torch.from_numpy(embeddings).to(self.device)
            valid_mask = ref_embeddings.norm(dim=1) > 0.1

            self._embedding_model = model
            self._ref_embeddings = ref_embeddings
            self._valid_mask = valid_mask
            self._embedding_is_dinov2 = False
            self._embedding_transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])
            print(f"[CLASSIFIER] Loaded {ref_embeddings.shape[0]} EfficientNet reference embeddings")
        except Exception as e:
            print(f"[CLASSIFIER] Failed to load EfficientNet: {e}")

    @property
    def mode(self) -> str:
        """Returns 'dinov2_supervised', 'dinov2_embedding', 'efficientnet_embedding', 'dual', or 'none'."""
        return self._mode

    def classify(
        self, crops: list, batch_size: int = 64
    ) -> list[tuple[int, float]]:
        """Classify cropped product images.

        Returns list of (category_id, confidence) for each crop.
        """
        if not crops or self._mode == "none":
            return [(0, 0.0)] * len(crops) if crops else []

        all_results = []
        for batch_start in range(0, len(crops), batch_size):
            batch_crops = crops[batch_start : batch_start + batch_size]
            batch_results = self._classify_batch(batch_crops)
            all_results.extend(batch_results)

        return all_results

    def _classify_batch(self, crops: list) -> list[tuple[int, float]]:
        """Classify a single batch of crops."""
        if self._mode == "dual":
            return self._classify_dual(crops)
        elif self._mode == "dinov2_supervised":
            return self._classify_supervised(crops)
        else:
            # dinov2_embedding or efficientnet_embedding
            return self._classify_embedding(crops)

    def _classify_supervised(self, crops: list) -> list[tuple[int, float]]:
        """Direct classification via DINOv2 + linear head."""
        batch = torch.stack([self._supervised_transform(c) for c in crops]).to(self.device)

        with torch.no_grad():
            features = self._supervised_model(batch)
            logits = self._supervised_head(features)
            probs = F.softmax(logits, dim=1)

        results = []
        for i in range(len(crops)):
            best_idx = probs[i].argmax().item()
            best_prob = probs[i, best_idx].item()
            results.append((best_idx, best_prob))

        return results

    def _classify_embedding(self, crops: list) -> list[tuple[int, float]]:
        """Classification via temperature-scaled embedding similarity."""
        batch = torch.stack([self._embedding_transform(c) for c in crops]).to(self.device)

        with torch.no_grad():
            embeddings = self._embedding_model(batch)
            embeddings = F.normalize(embeddings, dim=1)

        similarities = embeddings @ self._ref_embeddings.T
        similarities[:, ~self._valid_mask] = float("-inf")

        probs = F.softmax(similarities / TEMPERATURE, dim=1)

        results = []
        for i in range(len(crops)):
            best_idx = probs[i].argmax().item()
            best_prob = probs[i, best_idx].item()
            results.append((best_idx, best_prob))

        return results

    def _classify_dual(self, crops: list) -> list[tuple[int, float]]:
        """Dual-mode: combine supervised + embedding probabilities.

        Combined: 0.6 * supervised_prob + 0.4 * embedding_prob
        """
        # Supervised probabilities
        sup_batch = torch.stack([self._supervised_transform(c) for c in crops]).to(self.device)
        with torch.no_grad():
            sup_features = self._supervised_model(sup_batch)
            sup_logits = self._supervised_head(sup_features)
            sup_probs = F.softmax(sup_logits, dim=1)

        # Embedding probabilities
        emb_batch = torch.stack([self._embedding_transform(c) for c in crops]).to(self.device)
        with torch.no_grad():
            emb_features = self._embedding_model(emb_batch)
            emb_features = F.normalize(emb_features, dim=1)

        similarities = emb_features @ self._ref_embeddings.T
        similarities[:, ~self._valid_mask] = float("-inf")
        emb_probs = F.softmax(similarities / TEMPERATURE, dim=1)

        # Ensure both have the same number of classes
        n_sup = sup_probs.shape[1]
        n_emb = emb_probs.shape[1]
        n_classes = max(n_sup, n_emb)

        if n_sup < n_classes:
            sup_probs = F.pad(sup_probs, (0, n_classes - n_sup))
        if n_emb < n_classes:
            emb_probs = F.pad(emb_probs, (0, n_classes - n_emb))

        # Weighted combination
        combined = 0.6 * sup_probs + 0.4 * emb_probs

        results = []
        for i in range(len(crops)):
            best_idx = combined[i].argmax().item()
            best_prob = combined[i, best_idx].item()
            results.append((best_idx, best_prob))

        return results
