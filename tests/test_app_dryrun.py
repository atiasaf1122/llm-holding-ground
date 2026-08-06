"""The hand-off: what ``dryrun`` leaves on disk is what the dashboard reads.

The two ends of this are tested apart -- the CLI against its exit codes, the app
against frames a fixture built -- and the seam between them is where the offline
rehearsal was broken: the dry run kept its prices in memory, so it produced
artefacts the dashboard rejected, and nothing failed.

So both tests here run the real command and then the real page, headless, on CPU
with no daemon. They assert only what the seam promises: that the rehearsal draws,
and that an empty directory says so instead.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from council.cli import EXIT_OK, main
from council.config import PROJECT_ROOT, get_settings
from helpers_pipeline import MODELS, START, TICKERS

pytest.importorskip("streamlit", reason="the dashboard needs the `app` extra")

from streamlit.testing.v1 import AppTest

DASHBOARD = PROJECT_ROOT / "src" / "council" / "app" / "dashboard.py"

TIMEOUT = 120.0
"""Seconds. The finished-run page scores every arm on first draw, which is slower
than the three seconds AppTest allows by default."""

END = "2022-02-04"
"""A month of sessions: enough to clear the warm-up below and leave real decisions
behind it, short enough that a dry run inside a test finishes in seconds."""


@pytest.fixture
def configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the process-wide settings at an empty run directory.

    Through the environment and the settings cache, because that is the path the
    dashboard itself takes: it calls ``get_settings()`` rather than accepting one.
    """
    monkeypatch.setenv("COUNCIL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("COUNCIL_TICKERS", json.dumps(list(TICKERS)))
    monkeypatch.setenv("COUNCIL_AGENT_MODELS", json.dumps(list(MODELS)))
    monkeypatch.setenv("COUNCIL_LOOKBACK_DAYS", "5")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture
def rehearsed(configured: Path) -> Path:
    """A finished run in the configured directory, from the documented dry run."""
    code = main(
        [
            "dryrun",
            "--tickers",
            *TICKERS,
            "--models",
            *MODELS,
            "--start",
            START.isoformat(),
            "--end",
            END,
            "--data-dir",
            str(configured),
            "--log-level",
            "ERROR",
        ]
    )
    assert code == EXIT_OK
    return configured


def _run(path: Path) -> AppTest:
    app = AppTest.from_file(str(DASHBOARD), default_timeout=TIMEOUT)
    assert path.is_dir()
    return app.run()


def test_a_directory_with_no_run_in_it_says_so_rather_than_drawing_empty_axes(
    configured: Path,
) -> None:
    app = _run(configured)

    assert not app.exception
    assert "No run to read yet" in [header.value for header in app.header]


def test_the_page_draws_a_dry_run_without_raising(rehearsed: Path) -> None:
    # The dry run is what the README tells a reader to do before opening this page,
    # so a dry run whose artefacts the dashboard cannot read is a broken promise
    # rather than an inconvenience.
    app = _run(rehearsed)

    assert not app.exception
    assert "No run to read yet" not in [header.value for header in app.header]
    assert len(app.dataframe) > 0
