"""A finished run on disk, and a headless page pointed at it.

The two panels this supports -- assembly and layout -- hold no arithmetic, which
is why they had no tests. What they do hold is which frame each panel is handed
and whether a control reaches it, and those are exactly the defects a unit test
over a transform cannot see: a page where the equity curves answer one question
and the four tables under them answer another raises nothing and draws cleanly.

``streamlit.testing.v1.AppTest`` runs the script headless on CPU with no server,
so this costs a second and needs no GPU.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from council.app import dashboard
from council.app.artefacts import DASHBOARD_COLUMNS
from council.config import Settings, get_settings
from council.data.prices import synthetic_prices, write_prices
from council.debate.compositions import Composition, balanced_design
from council.evaluation.frames import NO_COMPOSITION, NO_FAILURE
from council.planning import TREATMENT_ARMS

DASHBOARD = Path(str(dashboard.__file__))
"""The script `streamlit run` is pointed at, located through the import rather
than rebuilt from the project root -- a path spelled out here would keep
resolving after the module moved, and the tests would silently drive a file that
is no longer the page."""

SESSIONS = 12
"""Sessions the fixture run covers. Long enough for the engine to form periods
and for the random baseline to have a turnover it can match, short enough that a
page renders in well under a second."""

DEBATED = (2, 5)
"""Which sessions held a conversation. Some rather than all, so a test can tell a
debate arm's curve apart from the control it falls back to elsewhere."""


def _row(
    *,
    on: object,
    ticker: str,
    model: str,
    persona: str,
    arm: str,
    round_index: int,
    composition: str,
    exposure: float,
    confidence: float,
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
        "rationale": f"{model} on {on}",
        "failure": NO_FAILURE,
    }


def decisions_frame(
    settings: Settings,
    *,
    days: list[Any],
    compositions: tuple[Composition, ...],
    arms: tuple[str, ...],
) -> pd.DataFrame:
    """A run with every arm, over the seats the configured design actually names.

    Exposures alternate so the control turns over, and a debate round moves the
    seat far enough to clear ``shift_threshold`` -- otherwise every rate on the
    page would be zero and a panel that silently showed the wrong population
    would look the same as one that showed the right one.

    Each committee moves by a different amount, for the same reason: a design in
    which every committee reached the same answer would make the pooled figure
    and any single committee's figure identical, and a scope control that reached
    nothing would pass.
    """
    rows: list[dict[str, Any]] = []
    seats = sorted(
        {(seat.model, seat.persona.name) for table in compositions for seat in table.seats}
    )
    for index, day in enumerate(days):
        opening = 0.5 if index % 2 == 0 else -0.5
        for ticker in settings.tickers:
            for model, persona in seats:
                rows.append(
                    _row(
                        on=day, ticker=ticker, model=model, persona=persona,
                        arm="independent", round_index=0, composition=NO_COMPOSITION,
                        exposure=opening, confidence=0.9,
                    )
                )
            if index not in DEBATED:
                continue
            for rank, table in enumerate(compositions):
                for arm_rank, arm in enumerate(arms):
                    for offset, seat in enumerate(table.seats):
                        moved = (rank + arm_rank + offset) % 3 == 0
                        final = -opening if moved else opening * 0.9
                        for round_index, exposure in ((0, opening), (1, final)):
                            rows.append(
                                _row(
                                    on=day, ticker=ticker, model=seat.model,
                                    persona=seat.persona.name, arm=arm,
                                    round_index=round_index,
                                    composition=table.identifier,
                                    exposure=exposure, confidence=0.9,
                                )
                            )
    return pd.DataFrame(rows, columns=list(DASHBOARD_COLUMNS))


def write_run(
    *, arms: tuple[str, ...] | None = None, committees: int | None = None
) -> Settings:
    """Write both artefacts a finished run leaves behind, where the page will look.

    Reads :func:`council.config.get_settings`, so the caller has already pointed
    ``COUNCIL_DATA_DIR`` at a directory of its own -- the same settings object
    the page under test will resolve.

    Args:
        arms: which treatment arms the run holds. Defaults to all three; a
            shorter list is how a test asks for the partial run the panels each
            have an empty case for.
        committees: how many of the design's committees ran. Defaults to all.
    """
    settings = get_settings()
    prices = synthetic_prices(
        tickers=settings.tickers, start=settings.start, sessions=SESSIONS, seed=settings.seed
    )
    write_prices(prices, settings.prices_path)

    days = sorted({value.date() for value in pd.to_datetime(prices["date"])})
    design = balanced_design(models=settings.agent_models)
    frame = decisions_frame(
        settings,
        days=days,
        compositions=design[: committees if committees is not None else len(design)],
        arms=arms if arms is not None else tuple(str(arm) for arm in TREATMENT_ARMS),
    )
    frame.to_parquet(settings.decisions_path, index=False)
    return settings


def page() -> AppTest:
    """A headless run of the dashboard script, already executed.

    ``from_file`` rather than ``from_function``: the script under test is the
    file a reader runs with ``streamlit run``, and driving ``main`` directly
    would leave the one line that calls it -- and any import the script does and
    the module does not -- unexercised.
    """
    return AppTest.from_file(str(DASHBOARD), default_timeout=60).run()


def run_dir(tmp_path: Path, monkeypatch: Any) -> Iterator[Path]:
    """A private directory the page reads, with every cache dropped either side.

    ``st.cache_data`` and ``get_settings`` both outlive one ``AppTest`` -- which
    is the point of a cache and a trap for a test, since the second test in a
    file would otherwise read the first test's artefacts.
    """
    monkeypatch.setenv("COUNCIL_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    st.cache_data.clear()
    yield tmp_path
    get_settings.cache_clear()
    st.cache_data.clear()


def captions(app: AppTest) -> str:
    """Every caption on the page, joined -- what the reader is told in small print."""
    return "\n".join(element.value for element in app.caption)


def warnings(app: AppTest) -> str:
    return "\n".join(element.value for element in app.warning)


def successes(app: AppTest) -> str:
    return "\n".join(element.value for element in app.success)


def selector(app: AppTest, key: str) -> Any:
    """One named selectbox, found by its key rather than by its position.

    Position would silently move the moment a panel gained a control, and the
    test would then drive a different widget while still passing.
    """
    for element in app.selectbox:
        if element.key == key:
            return element
    raise AssertionError(f"no selectbox keyed {key!r}")


def frame_named(app: AppTest, *, has: str) -> pd.DataFrame:
    """The first rendered dataframe carrying a given column."""
    for element in app.dataframe:
        frame = pd.DataFrame(element.value)
        if has in frame.columns:
            return frame
    raise AssertionError(f"no rendered table has a {has!r} column")
