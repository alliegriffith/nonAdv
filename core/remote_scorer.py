from __future__ import annotations

from typing import Any, Dict

import requests


def remote_wildguard_harm_score(
    sim_out: Dict[str, Any],
    *,
    base_url: str,
    scoring_target: str = "harmful_response",
    aggregate: str = "max",
    timeout: int = 180,
) -> float:
    response = requests.post(
        f"{base_url.rstrip('/')}/score",
        json={
            "sim_out": sim_out,
            "scoring_target": scoring_target,
            "aggregate": aggregate,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return float(response.json()["score"])