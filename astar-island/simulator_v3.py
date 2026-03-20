"""
Monte Carlo forward simulator v3 for the Astar Island Norse civilization challenge.

Key changes from v1/v2 based on divergence analysis against ground truth:
- Settlement survival: 86% in sim vs 32% in GT → much more fragile settlements
- Settlement death → mostly forest/empty (96%), rarely ruin (2.8% GT vs 50% sim)
- Fast forest reclamation of abandoned areas
- Mountain/ocean forced to deterministic distributions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ── Terrain codes ────────────────────────────────────────────────────────────

OCEAN = 10
PLAINS = 11
EMPTY = 0
SETTLEMENT = 1
PORT = 2
RUIN = 3
FOREST = 4
MOUNTAIN = 5

# Prediction class indices
NUM_CLASSES = 6
TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

# Immutable terrain (never changes during simulation)
IMMUTABLE = {OCEAN, MOUNTAIN}


# ── Settlement data ──────────────────────────────────────────────────────────

@dataclass
class Settlement:
    x: int
    y: int
    population: float = 1.0
    food: float = 1.0
    wealth: float = 0.0
    defense: float = 0.5
    has_port: bool = False
    alive: bool = True
    owner_id: int = 0
    tech_level: float = 0.0
    has_longship: bool = False
    age: int = 0  # Track settlement age for fragility

    def strength(self) -> float:
        return self.population * self.defense * (1.0 + 0.2 * self.tech_level)


# ── Default parameters ───────────────────────────────────────────────────────

DEFAULT_PARAMS: Dict[str, float] = {
    # v3: Calibrated for much higher settlement mortality
    # Target: Settlement survival ~32% over 50 years (was 86%)
    "winter_severity": 0.28,
    "faction_aggression": 0.18,
    "trade_activity": 0.5,
    "forest_growth_rate": 0.12,
    "expansion_rate": 0.40,
    "raid_range": 5.0,
    "food_per_forest": 0.45,
    "port_development_threshold": 0.5,
    "ruin_reclaim_rate": 0.20,
}


# ── Helper functions ─────────────────────────────────────────────────────────

def _coastal_mask(grid: np.ndarray) -> np.ndarray:
    """Boolean mask: True for land cells adjacent (4-connected) to ocean."""
    H, W = grid.shape
    ocean = grid == OCEAN
    padded = np.pad(ocean, 1, constant_values=False)
    coastal = np.zeros((H, W), dtype=bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        coastal |= padded[1 + dy : H + 1 + dy, 1 + dx : W + 1 + dx]
    land = ~ocean & (grid != MOUNTAIN)
    return coastal & land


def _count_adjacent(grid: np.ndarray, value: int) -> np.ndarray:
    """Count how many of the 8 neighbors equal `value`."""
    H, W = grid.shape
    match = (grid == value).astype(np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            sy = slice(max(0, -dy), min(H, H - dy))
            sx = slice(max(0, -dx), min(W, W - dx))
            ty = slice(max(0, dy), min(H, H + dy))
            tx = slice(max(0, dx), min(W, W + dx))
            count[ty, tx] += match[sy, sx]
    return count


def _manhattan_dist(y1: int, x1: int, y2: int, x2: int) -> int:
    return abs(y1 - y2) + abs(x1 - x2)


def _is_land(code: int) -> bool:
    return code not in (OCEAN, MOUNTAIN)


# ── Simulator ────────────────────────────────────────────────────────────────

class NorseSimulator:
    """
    v3 Monte Carlo simulator with calibrated settlement mortality.

    Key fixes:
    A. Settlement death → mostly forest/empty, rarely ruin
    B. Settlements much more fragile (higher winter loss, lower collapse threshold)
    C. New settlements very fragile (less resources from parent)
    D. Fast forest reclamation of ruins and abandoned sites
    E. Mountain/ocean forced to deterministic distributions
    """

    def __init__(
        self,
        initial_grid: Any,
        initial_settlements: List[Dict[str, Any]],
        params: Optional[Dict[str, float]] = None,
    ):
        self.initial_grid = np.asarray(initial_grid, dtype=np.int32)
        self.H, self.W = self.initial_grid.shape
        self.initial_settlements = initial_settlements
        self.params = {**DEFAULT_PARAMS, **(params or {})}

        # Precompute static maps
        self._ocean_mask = self.initial_grid == OCEAN
        self._mountain_mask = self.initial_grid == MOUNTAIN
        self._initial_coastal = _coastal_mask(self.initial_grid)

        # Precompute expansion candidate offsets (manhattan dist 2-4)
        self._expansion_offsets = [
            (dy, dx) for dy in range(-4, 5) for dx in range(-4, 5)
            if 2 <= abs(dy) + abs(dx) <= 4
        ]

    def _init_settlements(self) -> List[Settlement]:
        """Create settlement objects from initial state."""
        settlements = []
        for i, s in enumerate(self.initial_settlements):
            settlements.append(Settlement(
                x=s.get("x", 0),
                y=s.get("y", 0),
                population=s.get("population", 1.0),
                food=s.get("food", 1.0),
                wealth=s.get("wealth", 0.0),
                defense=s.get("defense", 0.5),
                has_port=s.get("has_port", False),
                alive=s.get("alive", True),
                owner_id=s.get("owner_id", i),
                tech_level=s.get("tech_level", 0.0),
                has_longship=s.get("has_longship", False),
                age=0,
            ))
        return settlements

    def _rebuild_grid(self, grid: np.ndarray, settlements: List[Settlement]) -> None:
        """Update grid to reflect current settlement states."""
        for s in settlements:
            if not (0 <= s.y < self.H and 0 <= s.x < self.W):
                continue
            if s.alive:
                grid[s.y, s.x] = PORT if s.has_port else SETTLEMENT
            # NOTE: v3 handles dead settlement terrain in _phase_winter directly
            # so we don't override it here

    def _count_forest_food(self, grid: np.ndarray, y: int, x: int) -> float:
        """Count food from adjacent forests."""
        food = 0.0
        fpf = self.params["food_per_forest"]
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.H and 0 <= nx < self.W and grid[ny, nx] == FOREST:
                    food += fpf
        return food

    def _count_adjacent_forest(self, grid: np.ndarray, y: int, x: int) -> int:
        """Count number of adjacent forest cells."""
        count = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.H and 0 <= nx < self.W and grid[ny, nx] == FOREST:
                    count += 1
        return count

    # ── Occupancy grid helpers ────────────────────────────────────────────────

    def _build_occupancy(self, settlements: List[Settlement]) -> np.ndarray:
        """Build a boolean grid marking cells within manhattan distance 1 of alive settlements."""
        occ = np.zeros((self.H, self.W), dtype=bool)
        for s in settlements:
            if not s.alive:
                continue
            y, x = s.y, s.x
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.H and 0 <= nx < self.W:
                        occ[ny, nx] = True
        return occ

    # ── Phase: Growth ────────────────────────────────────────────────────────

    def _phase_growth(
        self, grid: np.ndarray, settlements: List[Settlement], rng: np.random.Generator,
        coastal: np.ndarray = None,
    ) -> None:
        p = self.params
        if coastal is None:
            coastal = _coastal_mask(grid)

        for s in settlements:
            if not s.alive:
                continue

            s.age += 1

            # Food production: base is minimal, forests provide most food.
            forest_food = self._count_forest_food(grid, s.y, s.x)
            base_food = 0.10 + forest_food
            s.food += base_food * (0.5 + 0.8 * rng.random())

            # Food consumption scales with population (bigger = hungrier)
            consumption = s.population * 0.16
            s.food -= consumption

            # Food spoilage: cap food storage (can't stockpile indefinitely)
            s.food = min(s.food, 1.8 + s.population * 0.4)

            # Population growth: slower, only when food surplus
            if s.food > 1.5:
                growth = min(s.food * 0.06, 0.2) * (0.5 + rng.random())
                s.population += growth
                s.food -= growth * 0.8

            # Population cap: limited by local resources
            max_pop = 2.0 + forest_food * 2.0
            if s.population > max_pop:
                s.population = max_pop

            # Defense and tech slowly grow
            s.defense = min(s.defense + 0.015 * rng.random(), 1.5)
            s.tech_level = min(s.tech_level + 0.008 * rng.random(), 2.5)

            # Wealth decay — maintenance costs, corruption, etc.
            s.wealth *= 0.90

            # Port development: coastal + enough wealth (checked before degradation)
            if not s.has_port and coastal[s.y, s.x]:
                threshold = p["port_development_threshold"]
                if s.wealth > threshold and rng.random() < 0.15 * (1 + s.tech_level):
                    s.has_port = True
                    grid[s.y, s.x] = PORT

            # Longship development
            if not s.has_longship and s.has_port:
                if s.wealth > 0.5 and rng.random() < 0.1 * (1 + s.tech_level * 0.3):
                    s.has_longship = True

            # Port degradation
            if s.has_port and rng.random() < 0.01:
                s.has_port = False
                grid[s.y, s.x] = SETTLEMENT

        # ── Change C: New settlements very fragile ──
        # Expansion phase: new settlements get much less resources
        expand_rate = p["expansion_rate"] * 0.5
        expandable = [s for s in settlements if s.alive and s.population > 1.5 and s.food > 1.0]
        if expandable:
            occ = self._build_occupancy(settlements)
            for s in expandable:
                if rng.random() >= expand_rate:
                    continue

                # Find candidate land cells within range 2-4
                candidates = []
                for dy, dx in self._expansion_offsets:
                    ny, nx = s.y + dy, s.x + dx
                    if (
                        0 <= ny < self.H
                        and 0 <= nx < self.W
                        and grid[ny, nx] in (PLAINS, EMPTY, FOREST)
                        and not occ[ny, nx]
                    ):
                        candidates.append((ny, nx))

                if candidates:
                    ny, nx = candidates[rng.integers(len(candidates))]
                    # Change C: Much less resources for new settlements
                    # population * 0.2 (was 0.4), food * 0.2 (was 0.5), defense=0.15 (was 0.4)
                    new_sett = Settlement(
                        x=nx, y=ny,
                        population=s.population * 0.2,
                        food=s.food * 0.2,
                        wealth=s.wealth * 0.2,
                        defense=0.15,
                        has_port=False,
                        alive=True,
                        owner_id=s.owner_id,
                        tech_level=s.tech_level * 0.4,
                        age=0,  # New settlement starts at age 0
                    )
                    s.population *= 0.8
                    s.food *= 0.8
                    s.wealth *= 0.8
                    settlements.append(new_sett)
                    # Update occupancy for the new settlement
                    for ddy in range(-1, 2):
                        for ddx in range(-1, 2):
                            oy, ox = ny + ddy, nx + ddx
                            if 0 <= oy < self.H and 0 <= ox < self.W:
                                occ[oy, ox] = True
                    grid[ny, nx] = SETTLEMENT

    # ── Phase: Conflict ──────────────────────────────────────────────────────

    def _phase_conflict(
        self, grid: np.ndarray, settlements: List[Settlement], rng: np.random.Generator
    ) -> None:
        p = self.params
        aggression = p["faction_aggression"]
        base_range = p["raid_range"]

        alive = [s for s in settlements if s.alive]
        n_alive = len(alive)
        if n_alive < 2:
            return

        # Precompute coordinate and owner arrays for vectorized distance checks
        coords_y = np.array([s.y for s in alive], dtype=np.int32)
        coords_x = np.array([s.x for s in alive], dtype=np.int32)
        owners = np.array([s.owner_id for s in alive], dtype=np.int32)

        for i, attacker in enumerate(alive):
            if not attacker.alive:
                continue

            # Raid probability: higher when desperate (low food) or aggressive
            desperation = max(0.0, 1.0 - attacker.food) * 0.4
            raid_prob = aggression * 0.3 + desperation
            if rng.random() > raid_prob:
                continue

            # Find targets within range using vectorized distance
            raid_range = base_range * (2.5 if attacker.has_longship else 1.0)
            dists = np.abs(coords_y - attacker.y) + np.abs(coords_x - attacker.x)
            mask = (dists <= raid_range) & (dists > 0) & (owners != attacker.owner_id)
            target_indices = np.where(mask)[0]
            target_indices = [j for j in target_indices if alive[j].alive]
            if not target_indices:
                continue

            target = alive[target_indices[rng.integers(len(target_indices))]]

            # Resolve combat
            atk_str = attacker.strength() * (0.6 + 0.8 * rng.random())
            def_str = target.strength() * (0.6 + 0.8 * rng.random())

            if atk_str > def_str:
                loot_food = target.food * 0.4
                loot_wealth = target.wealth * 0.4
                attacker.food += loot_food
                attacker.wealth += loot_wealth
                target.food -= loot_food
                target.wealth -= loot_wealth
                target.population *= 0.65
                target.defense *= 0.6

                if rng.random() < 0.25 * aggression:
                    target.owner_id = attacker.owner_id
            else:
                attacker.population *= 0.85
                attacker.defense *= 0.8

    # ── Phase: Trade ─────────────────────────────────────────────────────────

    def _phase_trade(
        self, grid: np.ndarray, settlements: List[Settlement], rng: np.random.Generator
    ) -> None:
        p = self.params
        trade_act = p["trade_activity"]
        if trade_act < 0.05:
            return

        ports = [s for s in settlements if s.alive and s.has_port]

        for port in ports:
            fish_bonus = 0.21 * (0.5 + rng.random())
            port.food += fish_bonus
            port.wealth += 0.035 * (0.5 + rng.random())

        if len(ports) < 2:
            return

        trade_range = 15.0

        for i, port_a in enumerate(ports):
            for port_b in ports[i + 1 :]:
                dist = _manhattan_dist(port_a.y, port_a.x, port_b.y, port_b.x)
                if dist > trade_range:
                    continue
                trade_prob = trade_act * 0.5
                if port_a.owner_id == port_b.owner_id:
                    trade_prob *= 1.5
                if rng.random() > trade_prob:
                    continue

                trade_value = 0.15 * trade_act * (0.5 + rng.random())
                port_a.wealth += trade_value
                port_b.wealth += trade_value
                port_a.food += trade_value * 0.5
                port_b.food += trade_value * 0.5

                tech_diff = abs(port_a.tech_level - port_b.tech_level)
                if port_a.tech_level > port_b.tech_level:
                    port_b.tech_level += tech_diff * 0.08
                else:
                    port_a.tech_level += tech_diff * 0.08

    # ── Phase: Winter ────────────────────────────────────────────────────────
    # Change A: Settlement death → mostly forest/empty, rarely ruin
    # Change B: Settlements much more fragile

    def _phase_winter(
        self, grid: np.ndarray, settlements: List[Settlement], rng: np.random.Generator
    ) -> None:
        p = self.params
        severity = p["winter_severity"]

        for s in settlements:
            if not s.alive:
                continue

            # ── Change B: Harsher winter food loss ──
            # severity * (0.65 + 0.55 * rng.random()) — tuned for ~32% 50yr survival
            food_loss = severity * (0.65 + 0.55 * rng.random()) * (1.0 + s.population * 0.15)
            s.food -= food_loss

            # Population attrition in harsh winters
            if s.food < 0:
                pop_loss = min(abs(s.food) * 0.4, s.population * 0.5)
                s.population -= pop_loss
                s.food = 0.0

            # Natural population decay (disease, old age, emigration)
            s.population *= (0.96 + 0.03 * rng.random())

            # ── Change B: 2% annual base death chance per settlement ──
            if rng.random() < 0.02:
                s.alive = False
                self._handle_settlement_death(grid, s, settlements, rng)
                continue

            # ── Change B: Settlement collapse — tuned thresholds ──
            # Collapse threshold: population < 0.35 (was 0.3)
            # Collapse probability: 0.45 * severity (was 0.4 * severity)
            if s.population < 0.35 or (s.food < 0.2 and rng.random() < 0.45 * severity):
                s.alive = False
                self._handle_settlement_death(grid, s, settlements, rng)
                continue

    def _handle_settlement_death(
        self, grid: np.ndarray, s: Settlement,
        settlements: List[Settlement], rng: np.random.Generator
    ) -> None:
        """
        Change A: When settlement dies, terrain outcome is:
        - 8% → Ruin (catastrophic collapse)
        - ~40% → Forest (if adjacent to forest, higher probability)
        - ~52% → Plains/Empty
        """
        adj_forest = self._count_adjacent_forest(grid, s.y, s.x)

        roll = rng.random()
        if roll < 0.08:
            # 8% chance: catastrophic → Ruin
            grid[s.y, s.x] = RUIN
        elif adj_forest >= 1 and roll < 0.08 + 0.45:
            # If adjacent to forest: 45% chance → Forest
            grid[s.y, s.x] = FOREST
        elif adj_forest == 0 and roll < 0.08 + 0.25:
            # No adjacent forest: 25% chance → Forest (lower)
            grid[s.y, s.x] = FOREST
        else:
            # Remainder → Plains/Empty
            grid[s.y, s.x] = PLAINS

        # Disperse population to nearby friendly settlements
        nearby_friendly = [
            other for other in settlements
            if other.alive
            and other.owner_id == s.owner_id
            and other is not s
            and abs(s.y - other.y) + abs(s.x - other.x) <= 5
        ]
        if nearby_friendly and s.population > 0:
            dispersed_pop = s.population * 0.5
            per_sett = dispersed_pop / len(nearby_friendly)
            for nf in nearby_friendly:
                nf.population += per_sett
        s.population = 0.0

    # ── Phase: Environment ───────────────────────────────────────────────────
    # Change D: Fast forest reclamation

    def _phase_environment(
        self, grid: np.ndarray, settlements: List[Settlement], rng: np.random.Generator,
        coastal: np.ndarray = None,
    ) -> None:
        p = self.params
        forest_rate = p["forest_growth_rate"]
        reclaim_rate = p["ruin_reclaim_rate"]
        if coastal is None:
            coastal = _coastal_mask(grid)

        # Find all ruins
        ruin_ys, ruin_xs = np.where(grid == RUIN)

        alive_settlements = [s for s in settlements if s.alive]

        # Precompute forest adjacency once for this phase
        forest_adj_count = _count_adjacent(grid, FOREST)

        # Register new ruins (age tracking reset in run())
        for ry, rx in zip(ruin_ys, ruin_xs):
            key = (int(ry), int(rx))
            if key not in self._ruin_age:
                self._ruin_age[key] = 0
            self._ruin_age[key] += 1
        # Clean up non-ruins
        for key in list(self._ruin_age.keys()):
            if grid[key[0], key[1]] != RUIN:
                del self._ruin_age[key]

        for ry, rx in zip(ruin_ys, ruin_xs):
            # Check if any nearby thriving settlement can reclaim
            reclaimed = False
            for s in alive_settlements:
                dist = abs(int(ry) - s.y) + abs(int(rx) - s.x)
                if dist <= 5 and s.alive:
                    strength_factor = min(s.population, 2.0) * 0.5
                    if rng.random() < reclaim_rate * strength_factor:
                        new_sett = Settlement(
                            x=int(rx), y=int(ry),
                            population=s.population * 0.2,
                            food=s.food * 0.2,
                            wealth=s.wealth * 0.1,
                            defense=0.15,  # Change C: weaker new settlements
                            has_port=bool(coastal[ry, rx]),
                            alive=True,
                            owner_id=s.owner_id,
                            tech_level=s.tech_level * 0.3,
                            age=0,
                        )
                        s.population *= 0.8
                        s.food *= 0.8
                        settlements.append(new_sett)
                        grid[ry, rx] = PORT if new_sett.has_port else SETTLEMENT
                        reclaimed = True
                        break

            if not reclaimed:
                # ── Change D: Fast ruin reclamation ──
                # 25% per year → forest if adjacent to forest, 15% → plains
                adj_forest = forest_adj_count[ry, rx]

                if adj_forest > 0 and rng.random() < 0.25:
                    grid[ry, rx] = FOREST
                elif rng.random() < 0.15:
                    grid[ry, rx] = PLAINS

        # ── Change D: Empty former-settlement sites near forest: 10% → forest per year ──
        # Forest grows on empty/plains adjacent to existing forest
        empty_mask = (grid == PLAINS) | (grid == EMPTY)
        grow_candidates = np.where(empty_mask & (forest_adj_count > 0))

        for y, x in zip(grow_candidates[0], grow_candidates[1]):
            # Change D: 10% per year for empty cells near forest (was ~0.12%)
            # Use higher rate for cells adjacent to multiple forests
            adj_count = min(forest_adj_count[y, x], 4)
            prob = 0.03 + 0.02 * adj_count  # 5% base + 2% per adj forest, up to ~11%
            if rng.random() < prob:
                grid[y, x] = FOREST

        # Forest natural decay/clearing
        forest_ys, forest_xs = np.where(grid == FOREST)
        sett_near = _count_adjacent(grid, SETTLEMENT) + _count_adjacent(grid, PORT)
        for fy, fx in zip(forest_ys, forest_xs):
            base_prob = 0.001  # Very slow natural decay
            if sett_near[fy, fx] > 0:
                base_prob += 0.002 * sett_near[fy, fx]
            if rng.random() < base_prob:
                grid[fy, fx] = PLAINS

    # ── Main simulation loop ─────────────────────────────────────────────────

    def run(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Run one 50-year simulation.

        Returns:
            2D array (H, W) of terrain codes after 50 years.
        """
        rng = np.random.default_rng(seed)
        grid = self.initial_grid.copy()
        settlements = self._init_settlements()
        self._ruin_age = {}  # Reset ruin age tracking

        # Place initial settlements on grid
        self._rebuild_grid(grid, settlements)

        for year in range(50):
            # Compute coastal mask once per year (used by growth + environment)
            coastal = _coastal_mask(grid)

            self._phase_growth(grid, settlements, rng, coastal=coastal)
            self._phase_conflict(grid, settlements, rng)
            self._phase_trade(grid, settlements, rng)
            self._phase_winter(grid, settlements, rng)
            self._phase_environment(grid, settlements, rng, coastal=coastal)

            # Sync grid with settlement states
            self._rebuild_grid(grid, settlements)

            # Compact settlement list every 10 years to prevent unbounded growth
            if year % 10 == 9:
                settlements = [s for s in settlements if s.alive]

        return grid

    def run_to_classes(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Run one simulation and return class indices (0-5).
        """
        grid = self.run(seed)
        class_grid = np.zeros_like(grid)
        for code, cls in TERRAIN_TO_CLASS.items():
            class_grid[grid == code] = cls
        return class_grid

    @staticmethod
    def run_monte_carlo(
        initial_grid: Any,
        initial_settlements: List[Dict[str, Any]],
        params: Optional[Dict[str, float]] = None,
        n_runs: int = 200,
        seeds: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """
        Run n_runs simulations and return probability distribution.

        Returns:
            (H, W, 6) float64 array of class probabilities.

        Change E: Mountain → [0,0,0,0,0,1], Ocean → [1,0,0,0,0,0] forced.
        Uses floor 0.0005 for immutable terrain instead of 0.01.
        """
        sim = NorseSimulator(initial_grid, initial_settlements, params)
        H, W = sim.H, sim.W

        if seeds is None:
            seeds = list(range(n_runs))

        # Accumulate counts
        counts = np.zeros((H, W, NUM_CLASSES), dtype=np.int32)

        for i, s in enumerate(seeds):
            class_grid = sim.run_to_classes(s)
            np.add.at(counts, (np.arange(H)[:, None], np.arange(W)[None, :], class_grid), 1)

        # Convert to probabilities with Jeffreys smoothing
        alpha = 0.5
        probs = (counts.astype(np.float64) + alpha) / (n_runs + NUM_CLASSES * alpha)

        # Tiny floor to prevent log(0)
        probs = np.maximum(probs, 1e-6)
        probs /= probs.sum(axis=2, keepdims=True)

        # ── Change E: Force mountain and ocean to deterministic distributions ──
        floor = 0.0005

        # Ocean cells: [1,0,0,0,0,0] (class 0 = empty/ocean/plains)
        ocean_mask = sim._ocean_mask
        probs[ocean_mask] = floor
        probs[ocean_mask, 0] = 1.0 - 5 * floor  # class 0 gets nearly all probability

        # Mountain cells: [0,0,0,0,0,1] (class 5 = mountain)
        mountain_mask = sim._mountain_mask
        probs[mountain_mask] = floor
        probs[mountain_mask, 5] = 1.0 - 5 * floor  # class 5 gets nearly all probability

        return probs


# ── Utility: parameter inference from observations ───────────────────────────

def infer_params_from_transitions(
    initial_grid: np.ndarray,
    observed_counts: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate hidden simulator parameters from observed transition counts.
    """
    H, W = initial_grid.shape
    params: Dict[str, float] = {}

    # Build per-initial-class transition counts
    trans = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for code, cls in TERRAIN_TO_CLASS.items():
        mask = initial_grid == code
        if mask.any():
            trans[cls] += observed_counts[mask].sum(axis=0)

    row_sums = trans.sum(axis=1)
    row_sums = np.maximum(row_sums, 1.0)
    trans_probs = trans / row_sums[:, None]

    # Settlement -> Ruin rate => winter_severity + faction_aggression
    sett_to_ruin = trans_probs[1, 3] if row_sums[1] > 10 else 0.3
    params["winter_severity"] = float(np.clip(sett_to_ruin * 1.5, 0.05, 0.95))
    params["faction_aggression"] = float(np.clip(sett_to_ruin * 2.0, 0.05, 0.95))

    # Settlement -> Port rate => trade_activity
    sett_to_port = trans_probs[1, 2] if row_sums[1] > 10 else 0.1
    port_survival = trans_probs[2, 2] if row_sums[2] > 5 else 0.3
    params["trade_activity"] = float(np.clip((sett_to_port + port_survival) * 1.2, 0.05, 0.95))

    # Empty -> Forest rate => forest_growth_rate
    empty_to_forest = trans_probs[0, 4] if row_sums[0] > 20 else 0.05
    params["forest_growth_rate"] = float(np.clip(empty_to_forest * 5.0, 0.02, 0.5))

    # Empty -> Settlement rate => expansion_rate
    empty_to_sett = trans_probs[0, 1] if row_sums[0] > 20 else 0.05
    params["expansion_rate"] = float(np.clip(empty_to_sett * 5.0, 0.05, 0.5))

    # Ruin -> Settlement rate => ruin_reclaim_rate
    ruin_to_sett = trans_probs[3, 1] if row_sums[3] > 5 else 0.1
    params["ruin_reclaim_rate"] = float(np.clip(ruin_to_sett * 2.0, 0.05, 0.5))

    # Port -> Ruin tells us about overall harshness
    port_to_ruin = trans_probs[2, 3] if row_sums[2] > 5 else 0.2
    params["winter_severity"] = float(np.clip(
        params["winter_severity"] + port_to_ruin * 0.5, 0.05, 0.95
    ))

    # Food per forest: hard to infer directly, use settlement survival as proxy
    sett_survival = trans_probs[1, 1] if row_sums[1] > 10 else 0.5
    params["food_per_forest"] = float(np.clip(sett_survival * 0.5, 0.1, 0.6))

    # Port development threshold: inverse of port formation rate
    params["port_development_threshold"] = float(np.clip(1.0 - sett_to_port * 3.0, 0.2, 0.8))

    # Raid range: proportional to aggression
    params["raid_range"] = float(np.clip(3.0 + params["faction_aggression"] * 5.0, 3.0, 8.0))

    return params


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create a simple test grid
    size = 40
    grid = np.full((size, size), PLAINS, dtype=np.int32)

    # Ocean border
    grid[0, :] = OCEAN
    grid[-1, :] = OCEAN
    grid[:, 0] = OCEAN
    grid[:, -1] = OCEAN

    # Some mountains
    grid[10, 10:15] = MOUNTAIN
    grid[20, 20:25] = MOUNTAIN

    # Some forests
    grid[5:8, 5:8] = FOREST
    grid[15:18, 25:28] = FOREST
    grid[25:28, 10:13] = FOREST

    # Initial settlements
    settlements = [
        {"x": 8, "y": 8, "has_port": False, "alive": True},
        {"x": 15, "y": 5, "has_port": False, "alive": True},
        {"x": 30, "y": 15, "has_port": True, "alive": True},
        {"x": 5, "y": 25, "has_port": False, "alive": True},
        {"x": 25, "y": 30, "has_port": False, "alive": True},
    ]

    params = {
        "winter_severity": 0.3,
        "faction_aggression": 0.4,
        "trade_activity": 0.5,
        "forest_growth_rate": 0.2,
        "expansion_rate": 0.25,
    }

    print("Running Monte Carlo simulation v3 (100 runs)...")
    probs = NorseSimulator.run_monte_carlo(grid, settlements, params, n_runs=100)
    print(f"Output shape: {probs.shape}")
    print(f"Sum check (should be ~1.0): {probs[20, 20].sum():.4f}")
    print(f"Min prob: {probs.min():.6f}")

    # Show class distribution for a few cells
    class_names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
    for y, x in [(8, 8), (20, 20), (5, 5), (0, 0), (10, 12)]:
        p = probs[y, x]
        top = np.argmax(p)
        print(f"  ({y:2d},{x:2d}): {class_names[top]:10s} ({p[top]:.2f}) | {' '.join(f'{v:.2f}' for v in p)}")

    # Settlement survival check
    print("\n--- v3 Settlement Mortality Check ---")
    sim = NorseSimulator(grid, settlements, params)
    survival_count = 0
    n_test = 100
    for seed in range(n_test):
        final_grid = sim.run(seed)
        # Check initial settlement positions
        alive = 0
        for s in settlements:
            if final_grid[s["y"], s["x"]] in (SETTLEMENT, PORT):
                alive += 1
        survival_count += alive

    total_setts = len(settlements) * n_test
    print(f"Settlement survival rate: {survival_count}/{total_setts} = {survival_count/total_setts*100:.1f}%")
    print(f"Target: ~32% (was 86% in v1)")

    print("\nDone.")
