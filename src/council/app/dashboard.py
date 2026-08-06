"""The page: what was declared, then what happened.

Layout only. Every number shown is computed by a tested function in
:mod:`council.app`; nothing here reshapes data, and nothing here calls a model or
regenerates anything -- the page reads two artefacts off disk and draws them.

This module assembles; :mod:`council.app.panels` draws. The declaration comes
first and is rendered even when there is no run, because a reader must see what
was fixed in advance before they see any number that could have been chosen
after the fact.

**One scope for the whole page.** The committee selector reaches every panel, not
only the curves. A control that redrew one panel and silently left the four below
it pooled would put two different populations on one page under one heading, with
nothing saying so.

    streamlit run src/council/app/dashboard.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from council.app.artefacts import (
    ArtefactStatus,
    Results,
    artefact_status,
    load_results,
)
from council.app.curves import POOLED_LABEL, CurveSet, build_curves, compositions_for
from council.app.panels import (
    DECLARED,
    calibration_panel,
    equity_panel,
    influence_panel,
    shift_panel,
    transcript_panel,
)
from council.app.preregistration import PreRegistration, read_preregistration
from council.config import PROJECT_ROOT, Settings, get_settings
from council.debate.compositions import balanced_design
from council.evaluation.aggregation import RULE_NAMES, RULES
from council.scoring import PRIMARY_RULE

Stamp = tuple[float, int]
"""``(mtime, size)`` of one artefact. See :func:`_stamp`."""


def _stamp(path: Path) -> Stamp:
    """What makes one version of an artefact different from the next.

    Generation is an overnight job and
    :meth:`council.agents.store.DecisionStore.consolidate` rewrites
    ``decisions.parquet`` in place, so a server left running across a rewrite
    would otherwise serve the old parquet forever: a browser refresh reruns the
    script, and the script hits a cache keyed only on a path that has not
    changed. Reading the stamp here makes a rewritten file a different key.
    """
    stat = path.stat()
    return (stat.st_mtime, stat.st_size)


@st.cache_data(show_spinner=False)
def _load(decisions_path: Path, prices_path: Path, stamps: tuple[Stamp, Stamp]) -> Results:
    # `stamps` is deliberately unread: it is in the signature to be part of the
    # cache key, which is the whole reason a rewritten artefact reaches the page.
    del stamps
    return load_results(decisions_path=decisions_path, prices_path=prices_path)


@st.cache_data(show_spinner="Scoring the arms...")
def _curves(
    decisions: pd.DataFrame,
    opens: pd.DataFrame,
    *,
    composition: str | None,
    models: tuple[str, ...],
    rule_name: str,
    cost_bps: float,
    rebalance_threshold: float,
    seed: int,
) -> CurveSet:
    return build_curves(
        decisions=decisions,
        opens=opens,
        compositions=compositions_for(composition, balanced_design(models=models)),
        rule=RULES[rule_name],
        cost_bps=cost_bps,
        rebalance_threshold=rebalance_threshold,
        seed=seed,
    )


# -- 1. what was declared ----------------------------------------------------


def declaration(preregistration: PreRegistration, settings: Settings) -> None:
    st.title("Council")
    st.caption(
        "Do language models hold their ground? A controlled study of persuasion "
        "in multi-agent committees, scored by the market. Not a trading system."
    )
    st.header("The question")
    st.markdown(preregistration.question)
    st.header(f"{DECLARED} primary comparison")
    st.info(preregistration.primary_comparison)
    left, right = st.columns(2)
    left.metric("shift threshold", f"{settings.shift_threshold:.2f}")
    right.metric("dispersion threshold", f"{settings.dispersion_threshold:.2f}")
    st.caption(
        "Both fixed in config.py before any debate ran. Exactly two numbers below "
        f"are declared, and both are marked `{DECLARED}` where they appear: the "
        f"equity comparison under `{PRIMARY_RULE}` aggregation pooled over every "
        "committee, and the shift rate against prior confidence. What makes the "
        "rest of this page exploratory is named rather than implied -- the "
        "committee scope, the aggregation rule, and the arm and round selectors "
        "inside the panels."
    )


def no_run(status: ArtefactStatus) -> None:
    """What to say when there is nothing to show, which is not the same as nothing to see."""
    st.header("No run to read yet")
    st.warning(
        "This dashboard only reads artefacts. It never calls a model and never "
        "regenerates a result, so with the files below absent there is nothing to "
        "plot -- and drawing empty axes would suggest a run that produced nothing."
    )
    for item in status.missing:
        st.markdown(f"**{item.label}** -- `{item.path}`")
        st.caption(item.produced_by)


# -- assembly ----------------------------------------------------------------


def _read_at(stamps: tuple[Stamp, Stamp]) -> None:
    """Which version of the artefacts the page is showing.

    Two runs of one configuration produce artefacts of the same shape, so without
    a timestamp a reader cannot tell this morning's result from last week's.
    """
    written = max(stamp[0] for stamp in stamps)
    st.caption(f"Artefacts last written {datetime.fromtimestamp(written):%Y-%m-%d %H:%M:%S}.")


def _scope(results: Results) -> str | None:
    """The committee every panel is drawn from, or ``None`` for the declared scope."""
    st.sidebar.header("Committee")
    choice = st.sidebar.selectbox("scope", (POOLED_LABEL, *results.compositions))
    st.sidebar.caption(
        f"`{POOLED_LABEL}` is the declared scope -- the balanced design read as "
        "one experiment. Choosing a single committee is exploratory, and it "
        "applies to every panel on the page rather than to the curves alone."
    )
    return None if choice is None or choice == POOLED_LABEL else str(choice)


def results_panels(results: Results, settings: Settings) -> None:
    """Everything below the declaration, once a run has been read."""
    composition = _scope(results)
    rule_name = st.sidebar.selectbox(
        "aggregation", RULE_NAMES, index=RULE_NAMES.index(PRIMARY_RULE)
    )
    st.sidebar.caption(
        f"The primary comparison is declared over `{PRIMARY_RULE}`. The other "
        "rules are exploratory."
    )
    if rule_name is None:
        return

    try:
        curve_set = _curves(
            results.decisions,
            results.opens,
            composition=composition,
            models=settings.agent_models,
            rule_name=rule_name,
            cost_bps=settings.total_cost_bps,
            rebalance_threshold=settings.rebalance_threshold,
            seed=settings.seed,
        )
    except ValueError as error:
        st.error(f"The curves cannot be built for this committee: {error}")
    else:
        equity_panel(curve_set, settings, composition=composition, rule_name=rule_name)

    # Every panel below reads the same narrowed run, so the whole page describes
    # one population rather than one panel describing a committee and four
    # describing the pool.
    scoped = results.scoped_to(composition)
    shift_panel(scoped, settings, composition=composition)
    calibration_panel(scoped, composition=composition)
    influence_panel(scoped, composition=composition)
    transcript_panel(scoped, composition=composition)


def main() -> None:
    st.set_page_config(page_title="Council", layout="wide")
    settings = get_settings()
    declaration(read_preregistration(PROJECT_ROOT / "README.md"), settings)

    status = artefact_status(
        decisions_path=settings.decisions_path, prices_path=settings.prices_path
    )
    if not status.is_ready:
        no_run(status)
        return

    stamps = (_stamp(settings.decisions_path), _stamp(settings.prices_path))
    _read_at(stamps)
    results = _load(settings.decisions_path, settings.prices_path, stamps)
    if results.is_empty:
        st.error(
            f"`{settings.decisions_path}` holds no rows. A run that stored nothing "
            "is a fault rather than a state to wait out: check the generation logs."
        )
        return
    results_panels(results, settings)


if __name__ == "__main__":
    main()
