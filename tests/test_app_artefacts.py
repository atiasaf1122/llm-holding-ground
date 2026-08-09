"""Reading a run off disk, and reporting its absence as an absence.

The failure this file is really about is the empty one: a dashboard that renders
a flat curve and a shift rate of zero when no run exists is publishing numbers
nobody produced.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from council.app import artefacts
from council.app.artefacts import (
    ARM_ORDER,
    MissingArtefactsError,
    Results,
    artefact_status,
    load_results,
    order_arms,
    require_columns,
)
from council.data.prices import synthetic_prices
from council.domain.signal import Arm
from helpers_app import COMMITTEE, frame_of, independent, stored


def write_prices(directory: Path, *, tickers: tuple[str, ...] = ("AAA", "BBB")) -> Path:
    path = directory / "prices.parquet"
    synthetic_prices(tickers=tickers, sessions=10).to_parquet(path, index=False)
    return path


def write_decisions(directory: Path, frame: pd.DataFrame) -> Path:
    path = directory / "decisions.parquet"
    frame.to_parquet(path, index=False)
    return path


# -- what is missing ---------------------------------------------------------


def test_both_files_are_reported_missing_when_nothing_has_been_run(tmp_path: Path) -> None:
    status = artefact_status(
        decisions_path=tmp_path / "decisions.parquet", prices_path=tmp_path / "prices.parquet"
    )

    assert not status.is_ready
    assert [item.label for item in status.missing] == ["decisions", "prices"]


def test_every_missing_file_says_what_produces_it(tmp_path: Path) -> None:
    status = artefact_status(
        decisions_path=tmp_path / "decisions.parquet", prices_path=tmp_path / "prices.parquet"
    )

    assert all(item.produced_by for item in status.missing)
    assert "consolidate" in status.missing[0].produced_by


def test_a_present_file_is_not_reported_missing(tmp_path: Path) -> None:
    prices_path = write_prices(tmp_path)

    status = artefact_status(decisions_path=tmp_path / "nothing.parquet", prices_path=prices_path)

    assert [item.label for item in status.missing] == ["decisions"]


def test_loading_without_a_run_raises_rather_than_returning_an_empty_frame(
    tmp_path: Path,
) -> None:
    with pytest.raises(MissingArtefactsError, match="no run to read"):
        load_results(
            decisions_path=tmp_path / "decisions.parquet",
            prices_path=tmp_path / "prices.parquet",
        )


# -- the columns the panels need ---------------------------------------------


def test_a_frame_without_rationale_is_refused_because_the_transcript_needs_it() -> None:
    frame = frame_of(stored()).drop(columns=["rationale"])

    with pytest.raises(ValueError, match="rationale"):
        require_columns(frame)


def test_every_absent_column_is_named_at_once() -> None:
    frame = frame_of(stored()).drop(columns=["rationale", "confidence"])

    with pytest.raises(ValueError, match="confidence, rationale"):
        require_columns(frame)


# -- what a loaded run holds -------------------------------------------------


def test_loading_pairs_the_decisions_with_the_opens_they_were_scored_against(
    tmp_path: Path,
) -> None:
    write_decisions(tmp_path, frame_of(independent(ticker="AAA")))
    write_prices(tmp_path)

    results = load_results(
        decisions_path=tmp_path / "decisions.parquet", prices_path=tmp_path / "prices.parquet"
    )

    assert results.tickers == ("AAA",)
    assert len(results.decisions) == 1


def test_the_price_panel_is_narrowed_to_the_tickers_the_run_actually_covers(
    tmp_path: Path,
) -> None:
    # Buy-and-hold is an equal-weight basket of whatever columns it is handed, so
    # an untraded ticker left in would compare the committee against a different
    # universe from the one it was asked about.
    write_decisions(tmp_path, frame_of(independent(ticker="AAA")))
    write_prices(tmp_path, tickers=("AAA", "BBB"))

    results = load_results(
        decisions_path=tmp_path / "decisions.parquet", prices_path=tmp_path / "prices.parquet"
    )

    assert list(results.opens.columns) == ["AAA"]


def test_a_decision_on_a_ticker_with_no_price_history_raises(tmp_path: Path) -> None:
    write_decisions(tmp_path, frame_of(independent(ticker="ZZZ")))
    write_prices(tmp_path, tickers=("AAA",))

    with pytest.raises(ValueError, match="no price history for ZZZ"):
        load_results(
            decisions_path=tmp_path / "decisions.parquet",
            prices_path=tmp_path / "prices.parquet",
        )


def test_an_empty_decisions_file_is_distinguishable_from_a_missing_one() -> None:
    results = Results(decisions=frame_of(), opens=pd.DataFrame())

    assert results.is_empty


def test_the_arms_come_back_in_the_order_the_domain_declares_them() -> None:
    results = Results(
        decisions=frame_of(
            stored(arm="debate_placebo"), stored(arm="debate"), independent()
        ),
        opens=pd.DataFrame(),
    )

    assert results.arms == ("independent", "debate", "debate_placebo")


def test_the_independent_arms_null_composition_is_not_a_committee() -> None:
    results = Results(
        decisions=frame_of(independent(), stored(composition=COMMITTEE)), opens=pd.DataFrame()
    )

    assert results.compositions == (COMMITTEE,)


def test_an_unrecognised_arm_is_kept_rather_than_hidden() -> None:
    # A frame this code does not understand must not be presented as a complete one.
    assert order_arms({"debate", "cross-examination"}) == ("debate", "cross-examination")


def test_the_declared_order_puts_the_placebo_next_to_the_debate_it_controls_for() -> None:
    assert ARM_ORDER.index("debate_placebo") - ARM_ORDER.index("debate") <= 2


def test_narrowing_to_one_committee_keeps_the_control_it_is_read_against() -> None:
    # The independent arm carries no composition. Filtering the column alone
    # would delete the control from every panel that shows a committee beside
    # it, and each panel would then describe a population no label named.
    results = Results(
        decisions=frame_of(
            independent(),
            stored(composition=COMMITTEE),
            stored(composition="uniform-momentum-bold"),
        ),
        opens=pd.DataFrame(),
    )

    scoped = results.scoped_to(COMMITTEE)

    assert scoped.compositions == (COMMITTEE,)
    assert "independent" in scoped.arms


def test_the_declared_scope_returns_the_run_unchanged() -> None:
    results = Results(decisions=frame_of(independent(), stored()), opens=pd.DataFrame())

    assert results.scoped_to(None) is results


def _arm_order_doc() -> str:
    """ARM_ORDER's attribute docstring, read from the source.

    A ``Final`` tuple carries no ``__doc__`` at runtime, so the prose has to be read
    off the module the way a next engineer reads it.
    """
    source = Path(artefacts.__file__).read_text(encoding="utf-8")
    return source.split("ARM_ORDER: Final", 1)[1].split('"""')[1]


def test_the_arm_order_docstring_does_not_claim_an_adjacency_the_enum_has_not() -> None:
    # It justified the declared order by saying the debate arm and its placebo sit
    # next to each other, so a reader would not have to hunt across a table for the
    # gap that is the finding. `Arm` declares independent, debate, rationale-only,
    # placebo, so they do not.
    assert ARM_ORDER.index(str(Arm.DEBATE_PLACEBO)) - ARM_ORDER.index(str(Arm.DEBATE)) != 1
    assert "next to each other" not in _arm_order_doc()
