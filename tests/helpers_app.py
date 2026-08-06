"""Stored-shape decision frames for the dashboard tests.

The evaluation tests build frames with the columns the analysis reads. The app
additionally reads what the agents said, so these helpers build the *stored*
shape -- the columns the parquet on disk carries -- and every test states only
the two or three fields its question is about.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from council.app.artefacts import DASHBOARD_COLUMNS
from council.evaluation.frames import NO_COMPOSITION, NO_FAILURE

DAY = date(2022, 1, 3)
NEXT_DAY = date(2022, 1, 4)

OPENING = 0
REBUTTAL = 1

COMMITTEE = "rotation-0"


def stored(
    *,
    on: date = DAY,
    ticker: str = "AAA",
    model: str = "alpha",
    persona: str = "momentum-bold",
    arm: str = "debate",
    round_index: int = OPENING,
    composition: str = COMMITTEE,
    exposure: float = 0.0,
    confidence: float = 0.5,
    rationale: str = "",
    failure: str = NO_FAILURE,
) -> dict[str, Any]:
    return {
        "decision_date": on,
        "ticker": ticker,
        "model": model,
        "persona": persona,
        "arm": arm,
        "round_index": round_index,
        "composition": composition,
        "exposure": exposure,
        "confidence": confidence,
        "rationale": rationale,
        "failure": failure,
    }


def independent(**overrides: Any) -> dict[str, Any]:
    """One row of the control arm, which has no committee and no second round."""
    return stored(arm="independent", composition=NO_COMPOSITION, **overrides)


def frame_of(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(DASHBOARD_COLUMNS))
