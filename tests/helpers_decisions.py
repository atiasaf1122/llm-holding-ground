"""Hand-built decision frames for the evaluation tests.

Every frame in these tests is written out by hand with an answer that can be
checked on paper. The helper exists only so that a test states the two or three
fields its question is about and stays silent about the other six.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from council.evaluation.frames import NO_FAILURE, REQUIRED_COLUMNS

DAY = date(2022, 3, 1)
NEXT_DAY = date(2022, 3, 2)

OPENING = 0
REBUTTAL = 1


def row(
    *,
    on: date = DAY,
    ticker: str = "AAPL",
    model: str = "alpha",
    persona: str = "momentum-bold",
    arm: str = "debate",
    round_index: int = OPENING,
    composition: str = "quad",
    exposure: float = 0.0,
    confidence: float = 0.5,
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
        "failure": failure,
    }


def frame_of(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(REQUIRED_COLUMNS))


def debate_pair(
    *,
    model: str,
    opening: float,
    closing: float,
    confidence: float = 0.5,
    persona: str = "momentum-bold",
    on: date = DAY,
    composition: str = "quad",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One agent's two rounds: what it said, and what it said after hearing peers."""
    common: dict[str, Any] = {
        "model": model,
        "persona": persona,
        "on": on,
        "composition": composition,
    }
    return (
        row(round_index=OPENING, exposure=opening, confidence=confidence, **common),
        row(round_index=REBUTTAL, exposure=closing, confidence=confidence, **common),
    )
