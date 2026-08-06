"""The page itself, driven headless.

Everything here is about *assembly* rather than arithmetic: which frame each
panel is handed, whether the sidebar reaches it, what the page says when an
artefact is missing, and whether a rewritten artefact reaches a running server.
None of those can be caught by a test over a transform, and each of them fails by
drawing a clean page rather than by raising.

``AppTest`` runs the script in this process with no browser and no server, so the
whole file is CPU-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from council.app.curves import POOLED_LABEL
from council.config import get_settings

pytest.importorskip("streamlit", reason="the dashboard needs the `app` extra")

from council.app.panels import DECLARED
from helpers_dashboard import (
    captions,
    frame_named,
    page,
    run_dir,
    selector,
    successes,
    warnings,
    write_run,
)

run_directory = pytest.fixture(run_dir)


@pytest.fixture
def full_run(run_directory: Path) -> Path:
    write_run()
    return run_directory


# -- it renders at all -------------------------------------------------------


def test_a_finished_run_renders_every_panel_without_raising(full_run: Path) -> None:
    app = page()

    assert not app.exception
    assert [element.value for element in app.main.header] == [
        "The question",
        f"{DECLARED} primary comparison",
        "Equity curves",
        "Shift rate against prior confidence",
        "Calibration",
        "Influence",
        "Read one debate",
    ]


def test_with_no_artefacts_the_page_names_the_files_rather_than_drawing_empty_axes(
    run_directory: Path,
) -> None:
    app = page()

    assert not app.exception
    assert "No run to read yet" in [element.value for element in app.header]
    assert "decisions" in "\n".join(element.value for element in app.markdown)


def test_a_missing_price_file_alone_is_still_no_run(run_directory: Path) -> None:
    write_run()
    get_settings().prices_path.unlink()

    app = page()

    assert "No run to read yet" in [element.value for element in app.header]
    assert "dryrun" in captions(app)


def test_the_declaration_is_drawn_even_when_there_is_nothing_to_plot(
    run_directory: Path,
) -> None:
    # A reader must see what was fixed in advance before any number that could
    # have been chosen after the fact -- including when there are no numbers.
    app = page()

    assert app.title[0].value == "Council"
    assert f"{DECLARED} primary comparison" in [element.value for element in app.header]


# -- the committee selector reaches every panel ------------------------------


def _shift_counts(app: Any) -> list[int]:
    return [int(value) for value in frame_named(app, has="shift_rate")["count"]]


def _coverage(app: Any) -> pd.DataFrame:
    return frame_named(app, has="conversations")


def test_choosing_a_committee_changes_every_panel_and_not_only_the_curves(
    full_run: Path,
) -> None:
    # The defect this replaces: the sidebar redrew the equity curves and left the
    # shift, calibration, influence and transcript panels pooled over all eight
    # committees, byte-identical, with nothing on the page saying so. One page,
    # two populations, unlabelled.
    pooled = page()
    before_shift = _shift_counts(pooled)
    before_equity = frame_named(pooled, has="sharpe")["sharpe"].tolist()

    narrowed = page()
    narrowed.sidebar.selectbox[0].select("rotation-1").run()

    assert not narrowed.exception
    assert _shift_counts(narrowed) != before_shift
    assert frame_named(narrowed, has="sharpe")["sharpe"].tolist() != before_equity


def test_every_panel_states_the_population_it_covers(full_run: Path) -> None:
    app = page()

    assert captions(app).count(f"Population: {POOLED_LABEL}.") == 4


def test_a_narrowed_page_names_the_committee_in_every_panel(full_run: Path) -> None:
    app = page()
    app.sidebar.selectbox[0].select("rotation-1").run()

    assert captions(app).count("Population: the rotation-1 committee only.") == 4


def test_the_control_survives_narrowing_so_the_committee_has_something_to_be_read_against(
    full_run: Path,
) -> None:
    app = page()
    app.sidebar.selectbox[0].select("rotation-1").run()

    assert "independent" in frame_named(app, has="shift_rate")["arm"].tolist()


# -- which number is the declared one ----------------------------------------


def test_the_pooled_equity_panel_is_marked_as_the_declared_comparison(full_run: Path) -> None:
    app = page()

    assert f"{DECLARED} primary comparison" in successes(app)
    assert "council evaluate" in successes(app)


def test_the_shift_panel_is_marked_as_the_declared_statistic(full_run: Path) -> None:
    # The page previously captioned everything below the equity curves
    # "exploratory", which labelled its own pre-registered statistic exploratory.
    app = page()

    assert f"{DECLARED} primary statistic" in successes(app)


def test_choosing_one_committee_demotes_the_equity_panel_to_exploratory(
    full_run: Path,
) -> None:
    app = page()
    app.sidebar.selectbox[0].select("rotation-1").run()

    assert "Exploratory cut, not the declared comparison" in warnings(app)
    assert f"{DECLARED} primary comparison" not in successes(app)


def test_a_non_declared_aggregation_rule_demotes_the_equity_panel_too(
    full_run: Path,
) -> None:
    app = page()
    app.sidebar.selectbox[1].select("median").run()

    assert "`median` aggregation" in warnings(app)


def test_the_declaration_names_the_exploratory_axes_rather_than_the_whole_page(
    full_run: Path,
) -> None:
    app = page()

    assert "committee scope" in captions(app)
    assert "aggregation rule" in captions(app)


# -- the sample the primary statistic is declared over -----------------------


def test_the_shift_table_shows_distinct_points_beside_the_observation_count(
    full_run: Path,
) -> None:
    # `count` counts a point once per seat per committee. Reading it as the
    # sample size of a statistic declared over decision points overstates the
    # evidence by that factor.
    app = page()
    table = frame_named(app, has="shift_rate")
    scored = table.loc[table["count"] > 0]

    assert not scored.empty
    assert (scored["points"] < scored["count"]).all()


# -- what each arm actually covered ------------------------------------------


def test_the_shift_panel_reports_what_each_arm_covered(full_run: Path) -> None:
    app = page()
    coverage = _coverage(app)

    assert set(coverage["arm"]) >= {"independent", "debate", "debate_placebo"}
    assert "unpaired" in coverage.columns


def test_arms_that_covered_different_points_are_flagged(run_directory: Path) -> None:
    write_run()
    settings = get_settings()
    decisions = pd.read_parquet(settings.decisions_path)
    # Drop the placebo arm's earliest debated day, which is what
    # council.debate.sweep does when no earlier day can donate an argument.
    earliest = decisions.loc[decisions["arm"] == "debate_placebo", "decision_date"].min()
    trimmed = decisions.loc[
        ~((decisions["arm"] == "debate_placebo") & (decisions["decision_date"] == earliest))
    ]
    trimmed.to_parquet(settings.decisions_path, index=False)

    app = page()

    assert "do not cover the same decision points" in warnings(app)


# -- the transcript header's two populations ---------------------------------


def test_the_transcript_metrics_name_the_seats_they_cover(full_run: Path) -> None:
    # The spread is over every opening view and the means over the seats that
    # spoke twice, so with a silent seat they describe different committees.
    app = page()
    labels = [element.label for element in app.metric]

    assert "opening dispersion (all seats)" in labels
    assert "committee before (speaking seats)" in labels
    assert "committee after (speaking seats)" in labels


# -- a partial run -----------------------------------------------------------


def test_a_run_with_only_the_control_says_so_in_each_panel_rather_than_drawing_nothing(
    run_directory: Path,
) -> None:
    write_run(arms=())

    app = page()
    spoken = warnings(app)

    assert not app.exception
    assert "No paired rounds in this run" in spoken
    assert "no debate rows" in spoken
    assert "nothing to read" in spoken


# -- the rounds a population can be asked about ------------------------------


def test_the_control_is_offered_one_round_and_a_debate_arm_two(full_run: Path) -> None:
    # The independent arm has one round by construction, and offering a second
    # would draw an empty report that reads as a run with nothing in it.
    app = page()
    assert selector(app, "calibration_round").options == ["0 - opening view"]

    selector(app, "calibration_arm").select("debate").run()

    assert selector(app, "calibration_round").options == [
        "0 - opening view",
        "1 - after the debate",
    ]


# -- a run that cannot be scored ---------------------------------------------


def test_a_decisions_file_with_no_rows_is_a_fault_rather_than_a_state_to_wait_out(
    run_directory: Path,
) -> None:
    write_run()
    settings = get_settings()
    decisions = pd.read_parquet(settings.decisions_path)
    decisions.iloc[:0].to_parquet(settings.decisions_path, index=False)

    app = page()

    assert "holds no rows" in "\n".join(element.value for element in app.error)


def test_a_run_by_models_the_design_does_not_seat_says_so_instead_of_drawing_a_flat_curve(
    run_directory: Path,
) -> None:
    # A flat equity curve reads as a committee that decided nothing, which is a
    # very different claim from a run scored against seats nobody filled.
    write_run()
    settings = get_settings()
    decisions = pd.read_parquet(settings.decisions_path)
    decisions["model"] = "a-model-nobody-configured"
    decisions.to_parquet(settings.decisions_path, index=False)

    app = page()
    spoken = "\n".join(element.value for element in app.error)

    assert "The curves cannot be built" in spoken
    assert "a-model-nobody-configured" in spoken


# -- the cache lets go of a rewritten artefact -------------------------------


def test_a_rewritten_decisions_file_reaches_a_page_that_has_already_been_opened(
    run_directory: Path,
) -> None:
    # Generation is an overnight job and the store rewrites decisions.parquet in
    # place. Keyed on the path alone, the cached loader served the old parquet
    # for the life of the server and a browser refresh changed nothing.
    write_run(arms=())
    before = page()
    assert "debate" not in frame_named(before, has="series")["series"].tolist()

    write_run()
    after = page()

    assert "debate" in frame_named(after, has="series")["series"].tolist()
