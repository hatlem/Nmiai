#!/usr/bin/env python3
"""
XGBoost prediction server. Reads initial grid + regime features,
outputs full W×H×6 probability tensor.

Usage: python3 xgb_predict.py <init_json> <transitions_json> <output_json>
"""
import json, sys
import numpy as np
import xgboost as xgb
from pathlib import Path
from train_xgb import extract_features, NC, cc

MODEL_DIR = Path(__file__).parent

def load_models():
    models = []
    for c in range(NC):
        m = xgb.XGBRegressor()
        m.load_model(str(MODEL_DIR / f"xgb_class{c}.json"))
        models.append(m)
    return models

def predict_grid(models, ig, H, W, transitions):
    """Predict full H×W×6 tensor."""
    # Extract features for all non-static cells
    cells = []
    coords = []
    for y in range(H):
        for x in range(W):
            raw = ig[y][x]
            if raw == 5:  # mountain
                continue
            if raw == 10:  # ocean
                continue
            feat = extract_features(ig, y, x, H, W, transitions)
            cells.append(feat)
            coords.append((y, x))

    X = np.array(cells, dtype=np.float32)

    # Predict all classes
    pred = np.zeros((len(cells), NC), dtype=np.float64)
    for c in range(NC):
        pred[:, c] = models[c].predict(X)

    # Floor + normalize
    pred = np.maximum(pred, 0.0005)
    pred = pred / pred.sum(axis=1, keepdims=True)

    # Build full tensor
    FL = 0.0005
    tensor = np.full((H, W, NC), FL, dtype=np.float64)
    for i, (y, x) in enumerate(coords):
        tensor[y, x] = pred[i]

    # Static cells
    for y in range(H):
        for x in range(W):
            if ig[y][x] == 5:  # mountain
                tensor[y, x] = [FL, FL, FL, FL, FL, 1 - 5*FL]
            elif ig[y][x] == 10:  # ocean
                tensor[y, x] = [1 - 5*FL, FL, FL, FL, FL, FL]

    return tensor.tolist()

def main():
    if len(sys.argv) < 4:
        print("Usage: xgb_predict.py <init_json> <transitions_json> <output_json>")
        sys.exit(1)

    init_path = sys.argv[1]
    trans_path = sys.argv[2]
    out_path = sys.argv[3]

    with open(init_path) as f:
        init_data = json.load(f)

    transitions = None
    if trans_path != "null":
        with open(trans_path) as f:
            transitions = json.load(f)

    models = load_models()

    H = init_data.get('map_height', init_data.get('height', 40))
    W = init_data.get('map_width', init_data.get('width', 40))

    results = {}
    seeds_count = init_data.get('seeds_count', 5)
    for si in range(seeds_count):
        if 'initial_states' in init_data:
            ig = init_data['initial_states'][si]['grid']
        elif 'initial_grid' in init_data:
            ig = init_data['initial_grid']
        else:
            continue

        tensor = predict_grid(models, ig, H, W, transitions)
        results[str(si)] = tensor

    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f"Predictions saved for {len(results)} seeds")

if __name__ == '__main__':
    main()
