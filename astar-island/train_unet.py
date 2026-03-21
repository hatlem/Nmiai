#!/usr/bin/env python3
"""
Train a lightweight U-Net to predict per-cell probability distributions
for Astar Island grids. Uses cached GT data from completed rounds.

Usage: python3 train_unet.py --cache_dir cache --output unet_weights.pt
"""
import json, os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NC = 6
# Map raw terrain codes to 0-5
TTC = {10:0, 11:0, 0:0, 1:1, 2:2, 3:3, 4:4, 5:5}

def cc(v):
    return TTC.get(v, 0)

# ── Model ─────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(min(8, ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, ch), ch)

    def forward(self, x):
        r = x
        x = F.gelu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.gelu(x + r)


class GridUNet(nn.Module):
    """Lightweight U-Net: 40x40 input → 40x40×6 probability output."""
    def __init__(self, in_ch=11, base=32, drop=0.3):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1),
                                   nn.GroupNorm(8, base), nn.GELU(), ResBlock(base))
        self.down1 = nn.Conv2d(base, base*2, 2, stride=2)  # 20x20
        self.enc2 = nn.Sequential(nn.GroupNorm(8, base*2), nn.GELU(), ResBlock(base*2))
        self.down2 = nn.Conv2d(base*2, base*4, 2, stride=2)  # 10x10
        self.enc3 = nn.Sequential(nn.GroupNorm(8, base*4), nn.GELU(), ResBlock(base*4))
        self.down3 = nn.Conv2d(base*4, base*8, 2, stride=2)  # 5x5
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.GroupNorm(8, base*8), nn.GELU(),
                                         ResBlock(base*8), nn.Dropout2d(drop), ResBlock(base*8))
        # Decoder
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = nn.Sequential(ResBlock(base*8), nn.Conv2d(base*8, base*4, 1))
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = nn.Sequential(ResBlock(base*4), nn.Conv2d(base*4, base*2, 1))
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = nn.Sequential(ResBlock(base*2), nn.Conv2d(base*2, base, 1))
        # Head
        self.head = nn.Sequential(nn.Conv2d(base, base, 1), nn.GELU(), nn.Conv2d(base, NC, 1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        logits = self.head(d1)
        p = F.softmax(logits, dim=1)
        p = p.clamp(min=0.005)
        return p / p.sum(dim=1, keepdim=True)


# ── Feature Engineering ───────────────────────────────────────────────────

def encode_grid(ig, H, W):
    """Encode initial grid to feature tensor (C, H, W)."""
    grid = torch.zeros(H, W, dtype=torch.long)
    for y in range(H):
        for x in range(W):
            grid[y, x] = cc(ig[y][x])

    features = []
    # 1. One-hot terrain (6 channels)
    one_hot = F.one_hot(grid, NC).permute(2, 0, 1).float()
    features.append(one_hot)

    # 2. Coastal map
    ocean = (grid == 0).float().unsqueeze(0)
    land = (grid > 0).float().unsqueeze(0)
    ocean_nb = F.max_pool2d(F.pad(ocean, [1,1,1,1], mode='constant', value=0), 3, stride=1)
    coastal = land * ocean_nb
    features.append(coastal)

    # 3. Food neighbors (farm count in 3x3)
    farm = (grid == 4).float().unsqueeze(0)
    farm_pad = F.pad(farm, [1,1,1,1], mode='constant', value=0)
    food = F.avg_pool2d(farm_pad, 3, stride=1) * 9.0
    features.append(food / 8.0)

    # 4. Settlement distance (iterative dilation)
    sett = ((grid == 1) | (grid == 2)).float().unsqueeze(0)
    dist = torch.zeros(1, H, W)
    reached = sett.clone()
    for d in range(1, 25):
        dilated = F.max_pool2d(F.pad(reached, [1,1,1,1], mode='constant', value=0), 3, stride=1)
        new = dilated * (1 - (dist > 0).float()) * (1 - sett)
        dist = dist + new * d
        reached = (reached + dilated).clamp(0, 1)
    features.append(dist / 25.0)

    # 5. Mountain distance
    mount = (grid == 5).float().unsqueeze(0)
    mdist = torch.zeros(1, H, W)
    reached = mount.clone()
    for d in range(1, 25):
        dilated = F.max_pool2d(F.pad(reached, [1,1,1,1], mode='constant', value=0), 3, stride=1)
        new = dilated * (1 - (mdist > 0).float()) * (1 - mount)
        mdist = mdist + new * d
        reached = (reached + dilated).clamp(0, 1)
    features.append(mdist / 25.0)

    # 6. Settlement density in 5x5
    sett_pad = F.pad(sett, [2,2,2,2], mode='constant', value=0)
    sett_density = F.avg_pool2d(sett_pad, 5, stride=1)
    features.append(sett_density)

    result = torch.cat(features, dim=0)  # (11, H, W)
    return result


# ── Data Loading ──────────────────────────────────────────────────────────

def load_data(cache_dir):
    """Load all GT data from cache."""
    samples = []
    init_files = sorted(glob.glob(os.path.join(cache_dir, 'r[0-9]*_init.json')))
    for init_file in init_files:
        rnum = os.path.basename(init_file).split('_')[0]  # e.g. 'r1'
        with open(init_file) as f:
            init_data = json.load(f)

        seeds_count = init_data.get('seeds_count', 5)
        H = init_data.get('map_height', 40)
        W = init_data.get('map_width', 40)

        for si in range(seeds_count):
            gt_file = os.path.join(cache_dir, f'{rnum}_gt_s{si}.json')
            if not os.path.exists(gt_file):
                continue
            with open(gt_file) as f:
                gt_data = json.load(f)

            ig = gt_data.get('initial_grid', init_data['initial_states'][si]['grid'] if 'initial_states' in init_data else None)
            gt = gt_data['ground_truth']
            if ig is None:
                continue

            feat = encode_grid(ig, H, W)
            target = torch.zeros(NC, H, W)
            for y in range(H):
                for x in range(W):
                    for c in range(NC):
                        target[c, y, x] = gt[y][x][c]

            samples.append((feat, target, f'{rnum}_s{si}'))

    print(f'Loaded {len(samples)} samples from {len(init_files)} rounds')
    return samples


# ── Scoring (match competition metric) ────────────────────────────────────

def score_prediction(pred, target, H=40, W=40):
    """Entropy-weighted KL divergence score (competition metric)."""
    total = 0.0
    count = 0
    for y in range(H):
        for x in range(W):
            gt = target[:, y, x].numpy()
            pr = pred[:, y, x].numpy()
            # Entropy of GT
            ent = -np.sum(gt * np.log(gt + 1e-10))
            if ent < 0.01:
                continue  # skip deterministic cells
            # KL divergence
            kl = np.sum(gt * np.log((gt + 1e-10) / (pr + 1e-10)))
            # Score = 100 * (1 - kl / log(6)) weighted by entropy
            score = max(0, 100 * (1 - kl / np.log(6)))
            total += score * ent
            count += ent
    return total / count if count > 0 else 0


# ── Training ──────────────────────────────────────────────────────────────

def train(samples, epochs=500, lr=1e-3, wd=1e-3, patience=60, device='cpu'):
    model = GridUNet(in_ch=11, base=32, drop=0.3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Leave-one-round-out: hold out last round for validation
    # Group by round
    round_ids = list(set(s[2].split('_')[0] for s in samples))
    round_ids.sort()
    val_round = round_ids[-1]  # hold out most recent round
    print(f'Validation round: {val_round} ({len([s for s in samples if s[2].startswith(val_round)])} seeds)')

    train_data = [(s[0].to(device), s[1].to(device)) for s in samples if not s[2].startswith(val_round)]
    val_data = [(s[0].to(device), s[1].to(device)) for s in samples if s[2].startswith(val_round)]

    best_val = float('inf')
    best_state = None
    no_improve = 0

    for ep in range(epochs):
        model.train()
        ep_loss = 0
        perm = torch.randperm(len(train_data))
        for i in perm:
            feat, target = train_data[i.item()]
            feat = feat.unsqueeze(0)
            target = target.unsqueeze(0)

            # Random augmentation: rotation + flip
            k = torch.randint(0, 4, (1,)).item()
            feat = torch.rot90(feat, k, [2, 3])
            target = torch.rot90(target, k, [2, 3])
            if torch.rand(1).item() > 0.5:
                feat = feat.flip(2)
                target = target.flip(2)
            if torch.rand(1).item() > 0.5:
                feat = feat.flip(3)
                target = target.flip(3)

            pred = model(feat)
            # KL div loss
            loss = F.kl_div(pred.log().clamp(min=-10), target, reduction='batchmean', log_target=False)
            # Spatial smoothness
            smooth = ((pred[:,:,1:,:] - pred[:,:,:-1,:]).pow(2).mean() +
                       (pred[:,:,:,1:] - pred[:,:,:,:-1]).pow(2).mean())
            loss = loss + 0.01 * smooth

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()

        sched.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for feat, target in val_data:
                pred = model(feat.unsqueeze(0))
                vl = F.kl_div(pred.log().clamp(min=-10), target.unsqueeze(0),
                              reduction='batchmean', log_target=False)
                val_loss += vl.item()
        val_loss /= max(len(val_data), 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if ep % 25 == 0:
            print(f'  Epoch {ep:3d}: train={ep_loss/len(train_data):.4f} val={val_loss:.4f} best={best_val:.4f}')

        if no_improve >= patience:
            print(f'  Early stop at epoch {ep}')
            break

    model.load_state_dict(best_state)
    return model


def evaluate(model, samples, device='cpu'):
    """Evaluate model with competition scoring metric."""
    model.eval()
    round_scores = {}
    with torch.no_grad():
        for feat, target, name in samples:
            pred = model(feat.unsqueeze(0).to(device))
            pred = pred[0].cpu()
            score = score_prediction(pred, target)
            rnd = name.split('_')[0]
            if rnd not in round_scores:
                round_scores[rnd] = []
            round_scores[rnd].append(score)
            print(f'  {name}: {score:.2f}')

    print('\nPer-round averages:')
    total = 0
    for rnd in sorted(round_scores.keys()):
        avg = np.mean(round_scores[rnd])
        total += avg
        print(f'  {rnd}: {avg:.2f}')
    print(f'  Overall: {total/len(round_scores):.2f}')


def export_onnx(model, output_path, H=40, W=40, in_ch=11):
    """Export model to ONNX for use in Node.js."""
    model.eval()
    dummy = torch.randn(1, in_ch, H, W)
    torch.onnx.export(model, dummy, output_path,
                       input_names=['features'], output_names=['probabilities'],
                       opset_version=14, dynamic_axes=None)
    size_kb = os.path.getsize(output_path) / 1024
    print(f'Exported to {output_path} ({size_kb:.0f} KB)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', default='cache')
    parser.add_argument('--output', default='unet_weights.pt')
    parser.add_argument('--onnx', default='unet_model.onnx')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    samples = load_data(args.cache_dir)
    if not samples:
        print('No data found!')
        return

    if args.eval_only:
        model = GridUNet(in_ch=11, base=32).to(device)
        model.load_state_dict(torch.load(args.output, map_location=device))
        evaluate(model, samples, device)
        return

    model = train(samples, epochs=args.epochs, lr=args.lr, device=device)

    # Save weights
    torch.save(model.state_dict(), args.output)
    print(f'\nWeights saved to {args.output}')

    # Evaluate on all data
    print('\n=== Evaluation ===')
    evaluate(model, samples, device)

    # Export ONNX
    export_onnx(model, args.onnx)


if __name__ == '__main__':
    main()
