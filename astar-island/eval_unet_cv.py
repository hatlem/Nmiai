#!/usr/bin/env python3
"""Leave-one-round-out CV to check if U-Net generalizes."""
import json, os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_unet import GridUNet, encode_grid, load_data, score_prediction, NC, cc

def train_holdout(samples, holdout_round, epochs=300, lr=1e-3, device='cpu'):
    """Train on all rounds except holdout, return model."""
    model = GridUNet(in_ch=11, base=32, drop=0.3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_data = [(s[0].to(device), s[1].to(device)) for s in samples if not s[2].startswith(holdout_round)]

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(train_data))
        for i in perm:
            feat, target = train_data[i.item()]
            feat = feat.unsqueeze(0)
            target = target.unsqueeze(0)
            k = torch.randint(0, 4, (1,)).item()
            feat = torch.rot90(feat, k, [2, 3])
            target = torch.rot90(target, k, [2, 3])
            if torch.rand(1).item() > 0.5:
                feat = feat.flip(2)
                target = target.flip(2)

            pred = model(feat)
            loss = F.kl_div(pred.log().clamp(min=-10), target, reduction='batchmean', log_target=False)
            smooth = ((pred[:,:,1:,:] - pred[:,:,:-1,:]).pow(2).mean() +
                       (pred[:,:,:,1:] - pred[:,:,:,:-1]).pow(2).mean())
            loss = loss + 0.01 * smooth
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    return model

def main():
    device = 'cpu'
    samples = load_data('cache')

    # Get unique round IDs
    round_ids = sorted(set(s[2].split('_')[0] for s in samples))
    print(f'Rounds: {round_ids}')

    results = {}
    for holdout in round_ids:
        print(f'\n=== Holdout: {holdout} ===')
        model = train_holdout(samples, holdout, epochs=300, device=device)

        model.eval()
        scores = []
        with torch.no_grad():
            for feat, target, name in samples:
                if not name.startswith(holdout):
                    continue
                pred = model(feat.unsqueeze(0).to(device))[0].cpu()
                s = score_prediction(pred, target)
                scores.append(s)
                print(f'  {name}: {s:.2f}')

        avg = np.mean(scores) if scores else 0
        results[holdout] = avg
        print(f'  Avg: {avg:.2f}')

    print('\n=== LEAVE-ONE-ROUND-OUT CV ===')
    for r in sorted(results.keys()):
        print(f'  {r}: {results[r]:.2f}')
    overall = np.mean(list(results.values()))
    print(f'  Overall CV: {overall:.2f}')
    print(f'  (Lookup-only CV floor was 68.91)')

if __name__ == '__main__':
    main()
