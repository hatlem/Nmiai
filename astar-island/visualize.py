"""
Astar Island — Local Visualization Tool

Renders initial maps, simulation observations, and predictions.
Usage:
    python visualize.py <JWT_TOKEN>              # Visualize active round
    python visualize.py <JWT_TOKEN> --round <ID>  # Specific round
    python visualize.py <JWT_TOKEN> --analysis     # Post-round ground truth comparison
"""

import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys
import argparse

BASE = "https://api.ainm.no"

# Terrain codes
OCEAN, PLAINS, EMPTY = 10, 11, 0
SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN = 1, 2, 3, 4, 5

TERRAIN_TO_CLASS = {
    OCEAN: 0, PLAINS: 0, EMPTY: 0,
    SETTLEMENT: 1, PORT: 2, RUIN: 3, FOREST: 4, MOUNTAIN: 5,
}

CLASS_NAMES = ["Empty/Ocean/Plains", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

# Color scheme — distinctive Norse-themed palette
TERRAIN_COLORS = {
    OCEAN:      "#1a5276",  # Deep blue
    PLAINS:     "#d4c896",  # Sandy beige
    EMPTY:      "#d4c896",
    SETTLEMENT: "#c0392b",  # Red
    PORT:       "#e67e22",  # Orange
    RUIN:       "#7f8c8d",  # Grey
    FOREST:     "#27ae60",  # Green
    MOUNTAIN:   "#6c3483",  # Purple
}

CLASS_COLORS = ["#d4c896", "#c0392b", "#e67e22", "#7f8c8d", "#27ae60", "#6c3483"]


def create_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    return session


def grid_to_image(grid, H, W):
    """Convert terrain grid to RGB image array."""
    img = np.zeros((H, W, 3))
    hex_to_rgb = lambda h: tuple(int(h.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))

    for y in range(H):
        for x in range(W):
            cell = grid[y][x]
            color = TERRAIN_COLORS.get(cell, TERRAIN_COLORS[EMPTY])
            img[y][x] = hex_to_rgb(color)
    return img


def plot_initial_states(detail):
    """Plot the initial map state for all seeds."""
    W, H = detail["map_width"], detail["map_height"]
    initial_states = detail["initial_states"]
    n_seeds = len(initial_states)

    fig, axes = plt.subplots(1, n_seeds, figsize=(4 * n_seeds, 5))
    if n_seeds == 1:
        axes = [axes]

    for i, state in enumerate(initial_states):
        grid = state["grid"]
        img = grid_to_image(grid, H, W)
        axes[i].imshow(img, interpolation='nearest')

        # Mark settlements
        for s in state["settlements"]:
            marker = '⚓' if s.get("has_port") else '🏠'
            axes[i].plot(s["x"], s["y"], 'w.', markersize=2)

        axes[i].set_title(f"Seed {i} ({len(state['settlements'])} settlements)", fontsize=10)
        axes[i].set_xlabel(f"{W}x{H}")
        axes[i].tick_params(labelsize=7)

    # Legend
    legend_entries = [
        mpatches.Patch(color=TERRAIN_COLORS[OCEAN], label="Ocean"),
        mpatches.Patch(color=TERRAIN_COLORS[PLAINS], label="Plains"),
        mpatches.Patch(color=TERRAIN_COLORS[SETTLEMENT], label="Settlement"),
        mpatches.Patch(color=TERRAIN_COLORS[PORT], label="Port"),
        mpatches.Patch(color=TERRAIN_COLORS[RUIN], label="Ruin"),
        mpatches.Patch(color=TERRAIN_COLORS[FOREST], label="Forest"),
        mpatches.Patch(color=TERRAIN_COLORS[MOUNTAIN], label="Mountain"),
    ]
    fig.legend(handles=legend_entries, loc='lower center', ncol=7, fontsize=8)
    fig.suptitle(f"Round {detail['round_number']} — Initial States", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    return fig


def plot_observation(obs, seed_idx, query_idx, initial_grid, H, W):
    """Plot a single simulation observation overlaid on the initial map."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Full map with viewport highlighted
    img = grid_to_image(initial_grid, H, W)
    ax1.imshow(img, interpolation='nearest')
    vp = obs["viewport"]
    rect = plt.Rectangle((vp["x"] - 0.5, vp["y"] - 0.5), vp["w"], vp["h"],
                          linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
    ax1.add_patch(rect)
    ax1.set_title("Initial map + viewport", fontsize=10)

    # Viewport observation (final state after 50 years)
    obs_grid = obs["grid"]
    vh, vw = len(obs_grid), len(obs_grid[0]) if obs_grid else 0
    obs_img = grid_to_image(obs_grid, vh, vw)
    ax2.imshow(obs_img, interpolation='nearest')
    ax2.set_title(f"Observation (year 50) — seed {seed_idx}, query {query_idx}", fontsize=10)

    legend_entries = [
        mpatches.Patch(color=TERRAIN_COLORS[k], label=v)
        for k, v in [(OCEAN, "Ocean"), (PLAINS, "Plains"), (SETTLEMENT, "Settlement"),
                      (PORT, "Port"), (RUIN, "Ruin"), (FOREST, "Forest"), (MOUNTAIN, "Mountain")]
    ]
    fig.legend(handles=legend_entries, loc='lower center', ncol=7, fontsize=8)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    return fig


def plot_all_observations(detail, observations_per_seed):
    """Plot all observations for each seed as overlaid heatmaps."""
    W, H = detail["map_width"], detail["map_height"]
    initial_states = detail["initial_states"]
    n_seeds = len(initial_states)

    fig, axes = plt.subplots(2, n_seeds, figsize=(4 * n_seeds, 9))
    if n_seeds == 1:
        axes = axes.reshape(-1, 1)

    for seed_idx in range(n_seeds):
        grid = initial_states[seed_idx]["grid"]
        img = grid_to_image(grid, H, W)

        # Top row: initial state with observation viewports
        axes[0][seed_idx].imshow(img, interpolation='nearest')
        obs_list = observations_per_seed.get(seed_idx, [])
        colors_cycle = ['yellow', 'cyan', 'magenta', 'lime', 'white', 'orange']
        for qi, obs in enumerate(obs_list):
            vp = obs["viewport"]
            c = colors_cycle[qi % len(colors_cycle)]
            rect = plt.Rectangle((vp["x"] - 0.5, vp["y"] - 0.5), vp["w"], vp["h"],
                                  linewidth=1.5, edgecolor=c, facecolor='none', linestyle='--', alpha=0.7)
            axes[0][seed_idx].add_patch(rect)
        axes[0][seed_idx].set_title(f"Seed {seed_idx}: {len(obs_list)} obs", fontsize=9)

        # Bottom row: frequency heatmap from observations
        freq = np.zeros((H, W, 6))
        count = np.zeros((H, W))
        for obs in obs_list:
            vp = obs["viewport"]
            for ry, row in enumerate(obs["grid"]):
                for rx, cell in enumerate(row):
                    y, x = vp["y"] + ry, vp["x"] + rx
                    if 0 <= y < H and 0 <= x < W:
                        cls = TERRAIN_TO_CLASS.get(cell, 0)
                        freq[y][x][cls] += 1
                        count[y][x] += 1

        # Show dominant observed class
        dominant = np.zeros((H, W, 3))
        hex_to_rgb = lambda h: tuple(int(h.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
        for y in range(H):
            for x in range(W):
                if count[y][x] > 0:
                    cls = np.argmax(freq[y][x])
                    dominant[y][x] = hex_to_rgb(CLASS_COLORS[cls])
                else:
                    dominant[y][x] = (0.15, 0.15, 0.15)  # Dark = unobserved

        axes[1][seed_idx].imshow(dominant, interpolation='nearest')
        axes[1][seed_idx].set_title(f"Observed dominant class", fontsize=9)

    fig.suptitle("Observations Overview", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_predictions(detail, session, round_id):
    """Plot submitted predictions (argmax + confidence)."""
    n_seeds = len(detail["initial_states"])
    W, H = detail["map_width"], detail["map_height"]

    resp = session.get(f"{BASE}/astar-island/my-predictions/{round_id}")
    if resp.status_code != 200:
        print(f"No predictions found: {resp.status_code}")
        return None

    predictions_raw = resp.json()
    # Normalize: API may return list or dict
    if isinstance(predictions_raw, list):
        predictions = {i: p for i, p in enumerate(predictions_raw)}
    else:
        predictions = predictions_raw

    fig, axes = plt.subplots(2, n_seeds, figsize=(4 * n_seeds, 9))
    if n_seeds == 1:
        axes = axes.reshape(-1, 1)

    hex_to_rgb = lambda h: tuple(int(h.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))

    for seed_idx in range(n_seeds):
        pred = predictions.get(seed_idx, predictions.get(str(seed_idx)))
        if pred is None:
            axes[0][seed_idx].set_title(f"Seed {seed_idx}: not submitted")
            axes[1][seed_idx].set_title(f"Seed {seed_idx}: not submitted")
            continue

        argmax = np.array(pred.get("argmax", []))
        confidence = np.array(pred.get("confidence", []))

        if argmax.size == 0:
            continue

        # Top: argmax prediction
        pred_img = np.zeros((H, W, 3))
        for y in range(min(H, argmax.shape[0])):
            for x in range(min(W, argmax.shape[1])):
                cls = int(argmax[y][x])
                pred_img[y][x] = hex_to_rgb(CLASS_COLORS[cls])
        axes[0][seed_idx].imshow(pred_img, interpolation='nearest')
        axes[0][seed_idx].set_title(f"Seed {seed_idx}: predicted class", fontsize=9)

        # Bottom: confidence heatmap
        if confidence.size > 0:
            im = axes[1][seed_idx].imshow(confidence, cmap='RdYlGn', vmin=0, vmax=1,
                                           interpolation='nearest')
            axes[1][seed_idx].set_title(f"Seed {seed_idx}: confidence", fontsize=9)
            plt.colorbar(im, ax=axes[1][seed_idx], fraction=0.046, pad=0.04)

    fig.suptitle("Submitted Predictions", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_analysis(detail, session, round_id):
    """Plot post-round analysis: ground truth vs prediction."""
    n_seeds = len(detail["initial_states"])
    W, H = detail["map_width"], detail["map_height"]

    hex_to_rgb = lambda h: tuple(int(h.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))

    fig, axes = plt.subplots(3, n_seeds, figsize=(4 * n_seeds, 13))
    if n_seeds == 1:
        axes = axes.reshape(-1, 1)

    for seed_idx in range(n_seeds):
        resp = session.get(f"{BASE}/astar-island/analysis/{round_id}/{seed_idx}")
        if resp.status_code != 200:
            axes[0][seed_idx].set_title(f"Seed {seed_idx}: no analysis")
            continue

        data = resp.json()
        ground_truth = np.array(data.get("ground_truth", []))
        your_argmax = np.array(data.get("your_argmax", []))
        kl_per_cell = np.array(data.get("kl_per_cell", []))

        # Row 0: Ground truth
        if ground_truth.size > 0:
            gt_img = np.zeros((H, W, 3))
            for y in range(min(H, ground_truth.shape[0])):
                for x in range(min(W, ground_truth.shape[1])):
                    cls = int(np.argmax(ground_truth[y][x])) if len(ground_truth.shape) == 3 else int(ground_truth[y][x])
                    gt_img[y][x] = hex_to_rgb(CLASS_COLORS[min(cls, 5)])
            axes[0][seed_idx].imshow(gt_img, interpolation='nearest')
            axes[0][seed_idx].set_title(f"Seed {seed_idx}: ground truth", fontsize=9)

        # Row 1: Your prediction argmax
        if your_argmax.size > 0:
            pred_img = np.zeros((H, W, 3))
            for y in range(min(H, your_argmax.shape[0])):
                for x in range(min(W, your_argmax.shape[1])):
                    cls = int(your_argmax[y][x])
                    pred_img[y][x] = hex_to_rgb(CLASS_COLORS[min(cls, 5)])
            axes[1][seed_idx].imshow(pred_img, interpolation='nearest')
            axes[1][seed_idx].set_title(f"Seed {seed_idx}: your prediction", fontsize=9)

        # Row 2: KL divergence heatmap (error)
        if kl_per_cell.size > 0:
            im = axes[2][seed_idx].imshow(kl_per_cell, cmap='hot', interpolation='nearest')
            axes[2][seed_idx].set_title(f"Seed {seed_idx}: KL divergence", fontsize=9)
            plt.colorbar(im, ax=axes[2][seed_idx], fraction=0.046, pad=0.04)

        score = data.get("score") or data.get("seed_score")
        if score is not None:
            axes[0][seed_idx].set_xlabel(f"Score: {score:.4f}", fontsize=8)

    fig.suptitle("Post-Round Analysis: Ground Truth vs Prediction", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def run_live_observations(session, detail, round_id):
    """Run simulations and show them interactively."""
    W, H = detail["map_width"], detail["map_height"]
    initial_states = detail["initial_states"]

    budget = session.get(f"{BASE}/astar-island/budget").json()
    remaining = budget["queries_max"] - budget["queries_used"]
    print(f"Budget: {budget['queries_used']}/{budget['queries_max']} ({remaining} remaining)")

    if remaining == 0:
        print("No queries remaining — showing existing data only.")
        return {}

    print("\nThis will NOT use queries. Showing initial states only.")
    print("To run observations, use agent.py and then visualize the results.")
    return {}


def main():
    parser = argparse.ArgumentParser(description="Astar Island Visualizer")
    parser.add_argument("token", help="JWT auth token")
    parser.add_argument("--round", dest="round_id", help="Specific round ID")
    parser.add_argument("--analysis", action="store_true", help="Show post-round analysis")
    parser.add_argument("--predictions", action="store_true", help="Show submitted predictions")
    parser.add_argument("--save", action="store_true", help="Save figures as PNG instead of showing")
    args = parser.parse_args()

    session = create_session(args.token)

    # Find round
    if args.round_id:
        detail = session.get(f"{BASE}/astar-island/rounds/{args.round_id}").json()
        round_id = args.round_id
    else:
        rounds = session.get(f"{BASE}/astar-island/rounds").json()
        active = next((r for r in rounds if r["status"] == "active"), None)
        if not active:
            print("No active round. Available rounds:")
            for r in rounds:
                print(f"  Round {r['round_number']}: {r['status']} — {r['id']}")
            if rounds:
                print(f"\nShowing latest round...")
                active = rounds[-1]
            else:
                return
        round_id = active["id"]
        detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()

    W, H = detail["map_width"], detail["map_height"]
    n_seeds = len(detail["initial_states"])
    print(f"Round {detail['round_number']}: {W}x{H}, {n_seeds} seeds, status={detail['status']}")

    figures = []

    # Always show initial states
    fig1 = plot_initial_states(detail)
    figures.append(("initial_states", fig1))

    # Show predictions if requested or available
    if args.predictions:
        fig_pred = plot_predictions(detail, session, round_id)
        if fig_pred:
            figures.append(("predictions", fig_pred))

    # Show analysis if requested and round is complete
    if args.analysis:
        fig_analysis = plot_analysis(detail, session, round_id)
        if fig_analysis:
            figures.append(("analysis", fig_analysis))

    # Save or show
    if args.save:
        for name, fig in figures:
            path = f"astar-island/viz_{name}_{detail['round_number']}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
