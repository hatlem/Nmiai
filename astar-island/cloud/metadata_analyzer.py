"""Extract hidden parameter estimates from settlement metadata."""

from typing import Any

import numpy as np


def extract_params(all_settlements: list[dict[str, Any]]) -> dict[str, float]:
    """From observed settlement metadata, estimate round characteristics.

    Args:
        all_settlements: list of settlement dicts from /simulate responses.
            Each has: population, food, wealth, defense, alive, has_port, owner_id

    Returns:
        dict with estimated parameters: survival_rate, avg_food, avg_population,
        avg_wealth, port_rate, faction_count, death_rate
    """
    if not all_settlements:
        return {
            "survival_rate": 0.5,
            "avg_food": 1.0,
            "avg_population": 1.0,
            "avg_wealth": 0.5,
            "port_rate": 0.0,
            "faction_count": 1,
            "death_rate": 0.5,
        }

    alive = [s for s in all_settlements if s.get("alive")]
    dead = [s for s in all_settlements if not s.get("alive")]

    total = len(alive) + len(dead)
    survival_rate = len(alive) / max(total, 1)

    if alive:
        avg_food = float(np.mean([s.get("food", 0) for s in alive]))
        avg_pop = float(np.mean([s.get("population", 0) for s in alive]))
        avg_wealth = float(np.mean([s.get("wealth", 0) for s in alive]))
        port_rate = sum(1 for s in alive if s.get("has_port")) / len(alive)
        faction_count = len(set(s.get("owner_id", 0) for s in alive))
    else:
        avg_food = 0.0
        avg_pop = 0.0
        avg_wealth = 0.0
        port_rate = 0.0
        faction_count = 0

    return {
        "survival_rate": survival_rate,
        "avg_food": avg_food,
        "avg_population": avg_pop,
        "avg_wealth": avg_wealth,
        "port_rate": port_rate,
        "faction_count": faction_count,
        "death_rate": 1.0 - survival_rate,
    }
