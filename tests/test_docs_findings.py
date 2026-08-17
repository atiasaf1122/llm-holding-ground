"""The write-up prints the numbers the shipped code produces.

``docs/findings.md`` is prose beside a stored artefact, which is the one place a
correction can be applied to the code and to ``docs/CLAIMS.md`` and still leave a
published table stating the superseded result. It happened once: ``cbf6a55``
replaced a bare ``distance >= threshold`` with
:func:`council.evaluation.threshold.meets`, rewrote C8-C11 and added D2-D5, and left
this document printing the pre-fix rates and reading conclusions off them.

So the table is recomputed here from the stored decisions rather than trusted, and
the caveats the register records are checked to appear where a reader meets the
number they qualify.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from council.config import PROJECT_ROOT
from council.domain.signal import Arm
from council.evaluation.frames import frame_to_rows
from council.evaluation.persuasion import shift_rate_by_confidence, shifts
from council.scoring import rows_in_arm

FINDINGS = PROJECT_ROOT / "docs" / "findings.md"
CLAIMS = PROJECT_ROOT / "docs" / "CLAIMS.md"
DECISIONS = PROJECT_ROOT / "docs" / "results" / "run-4models-2y" / "decisions.parquet"
"""The artefact the write-up reports. Published rather than left in ``data/`` so that
a reader can recompute the table, and so that this test can."""

PUBLISHED_ARMS: tuple[Arm, ...] = (Arm.DEBATE, Arm.DEBATE_PLACEBO, Arm.DEBATE_RATIONALE_ONLY)
"""The three columns of the shift-rate table, in the order it prints them."""

HEADING = "### Shift rate, by the confidence held before seeing any peer"

ROW = re.compile(r"^\|\s*(\d\.\d\d)\s")
"""A table row, keyed by the lower edge of the confidence band it reports. Stopping
at the whitespace after two decimals rather than at the label's dash, which is an en
dash in the document and has no business being pasted into a pattern."""

CELL = re.compile(r"(\d\.\d{3})\**\s*\((\d+)\)")
"""``rate (count)``, with the bold markers the placebo column wears ignored."""

MINUS = "\u2212"
"""The typographic minus the documents print negatives with; normalised to a hyphen
before any interval string is matched."""

Cells = list[tuple[str, int]]


@pytest.fixture(scope="module")
def published() -> pd.DataFrame:
    return pd.read_parquet(DECISIONS)


def published_table() -> dict[float, Cells]:
    """Every ``rate (count)`` in the shift-rate table, keyed by the band's lower edge."""
    document = FINDINGS.read_text(encoding="utf-8")
    assert HEADING in document, "the shift-rate table lost the heading this test finds it by"
    rows: dict[float, Cells] = {}
    for line in document.split(HEADING)[1].splitlines():
        match = ROW.match(line)
        if match is None:
            if rows:
                break
            continue
        rows[float(match.group(1))] = [(rate, int(count)) for rate, count in CELL.findall(line)]
    return rows


def recomputed(decisions: pd.DataFrame, arm: Arm) -> dict[float, tuple[str, int]]:
    """What the shipped code makes of the same artefact, keyed the same way."""
    report = shift_rate_by_confidence(shifts(rows_in_arm(decisions, arm), threshold=0.20))
    return {
        band.band.lower: (f"{band.shift_rate:.3f}", band.count)
        for band in report.bands
        if band.shift_rate is not None
    }


def test_the_published_shift_table_is_what_the_shipped_code_computes(
    published: pd.DataFrame,
) -> None:
    table = published_table()
    assert len(table) == 5, "the table lost or gained a confidence band"

    for column, arm in enumerate(PUBLISHED_ARMS):
        printed = {lower: cells[column] for lower, cells in table.items()}
        assert printed == recomputed(published, arm), arm


def test_the_pooled_rates_the_headline_rests_on_are_the_stored_ones(
    published: pd.DataFrame,
) -> None:
    """The three numbers a reader takes away, checked against the parquet.

    The banded table above is recomputed cell by cell, but the sentence under it
    quotes one pooled rate per arm and that is what gets repeated. It has its own
    way of drifting: a band added, dropped or reweighted changes the pooled figure
    without touching a single cell.
    """
    document = " ".join(FINDINGS.read_text(encoding="utf-8").split())
    for arm, printed in (
        (Arm.DEBATE, "debate **0.324**"),
        (Arm.DEBATE_RATIONALE_ONLY, "rationale-only **0.323**"),
        (Arm.DEBATE_PLACEBO, "placebo **0.383**"),
    ):
        moved = [shift.shifted for shift in shifts(rows_in_arm(published, arm), threshold=0.20)]
        assert f"{sum(moved) / len(moved):.3f}" in printed, arm
        assert printed in document, arm


# -- the caveats sit where the number they qualify does ----------------------------


def test_the_headline_is_stated_with_an_interval_rather_than_a_bare_difference() -> None:
    """A 6pp gap over 50 decision points is a claim about a distribution.

    The arms differ by more than this between confidence bands, and observations
    inside one decision point are not independent -- so a difference printed without
    the interval it survives is a number a reader cannot weigh.
    """
    section = FINDINGS.read_text(encoding="utf-8").split("## 1. The placebo moves")[1]
    body = " ".join(section.split("\n## ")[0].split())

    assert "95% CI" in body
    assert "[+3.31, +8.75]" in body
    assert "paired by decision point" in body


def test_the_headline_does_not_claim_the_mechanism_the_design_cannot_separate() -> None:
    """C8 supports "not the argument's content" and nothing about contradiction.

    The placebo peer argues about another day, so its argument is incoherent against
    the reader's data and not merely irrelevant. Reacting to being contradicted and
    failing to place an argument are different behaviours; no arm here separates
    them, and the register carries that as D8.
    """
    claims = CLAIMS.read_text(encoding="utf-8")
    c8 = claims.split("C8.")[1].split("C9.")[0]
    findings = FINDINGS.read_text(encoding="utf-8")

    assert "\nD8." in claims
    assert "incoherent" in " ".join(findings.split()).lower()
    assert "D8" in findings, "the write-up meets the caveat, so it carries the pointer"
    assert "reacting to contradiction" not in " ".join(c8.split()).lower()


def test_the_anchoring_finding_separates_convergence_from_mind_changing() -> None:
    """The one result that would read as its opposite if either half were dropped.

    Spread narrows most in the full debate arm while the shift rate is flat against
    rationale-only. Quoted alone, the first says peers persuade and the second says
    they do not.
    """
    body = " ".join(
        FINDINGS.read_text(encoding="utf-8")
        .split("## 2. Seeing peers' numbers")[1]
        .split("\n## ")[0]
        .split()
    )

    assert "+0.12pp" in body, "the shift-rate half"
    assert "1.264" in body and "0.703" in body, "the spread half"


def test_the_returns_table_is_marked_as_description_rather_than_evidence() -> None:
    """Fifty debated points of 1,002 cannot test whether debate changes returns.

    The four arms' returns agreeing to three decimal places is what a diluted
    comparison looks like, not what a null result looks like, and the difference is
    invisible unless the document says so.
    """
    body = " ".join(
        FINDINGS.read_text(encoding="utf-8")
        .split("## 6. Nobody made any money")[1]
        .split("\n## ")[0]
        .split()
    )

    assert "50 of 1,002" in body
    assert "not evidence" in body or "not a test" in body


def test_the_probe_does_not_order_the_models() -> None:
    """Three of four net rates are exactly zero and the fourth is one event.

    The earlier version of this claim ranked two models on a difference of zero
    events, which is D7. The corpus is the same size; the temptation is the same.
    """
    claims = CLAIMS.read_text(encoding="utf-8")
    c12 = " ".join(claims.split("C12.")[1].split("C13.")[0].split())
    section = " ".join(
        FINDINGS.read_text(encoding="utf-8")
        .split("## 4. On questions with a right answer")[1]
        .split("\n## ")[0]
        .split()
    )

    assert "does not order" in c12
    assert "does **not** order the models" in section
    assert "upper bound" in c12 and "upper bound" in section


def test_the_write_up_no_longer_calls_the_placebo_day_unrelated() -> None:
    assert "about a different day entirely" not in FINDINGS.read_text(encoding="utf-8")


def test_c3_states_the_balance_property_the_design_actually_has() -> None:
    # rotations() puts each model at each persona once; uniform_references() puts it
    # there once more. Across all eight committees the count is two, not one.
    c3 = " ".join(CLAIMS.read_text(encoding="utf-8").split("C3.")[1].split("C4.")[0].split())

    assert "at every persona exactly once" not in c3
    assert "the same number of times" in c3
    assert "uniform references" in c3


def test_the_superseded_defects_are_kept_rather_than_deleted() -> None:
    """D2-D7 describe classes of error, not one run's arithmetic.

    A float comparison on a grid, a control that is not inert, and an ordering built
    on one event are all live risks in the current design. Deleting them with the
    figures they attacked would delete the reason each check exists.
    """
    claims = CLAIMS.read_text(encoding="utf-8")

    assert "Superseded defects, retained for provenance" in claims
    for defect in ("D2", "D7"):
        assert defect in claims
    assert "docs/results/superseded/" in claims


# -- the confidence result, and the half of it that was withheld -------------------


def test_the_calibration_figures_are_the_ones_the_artefact_holds(
    published: pd.DataFrame,
) -> None:
    """Confidence against being right, recomputed rather than quoted.

    The largest sample in the study, and the one number here a reader is most likely
    to carry off and act on: a committee weighted by self-reported confidence is
    weighted by noise. So it is checked against the parquet like the shift table.
    """
    from council.evaluation.calibration import calibrate
    from council.evaluation.frames import forward_returns, forward_returns_lookup
    from council.scoring import _scoring_window

    prices = pd.read_parquet(DECISIONS.parent / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    opens = prices.pivot(index="date", columns="ticker", values="open")
    control = rows_in_arm(published, Arm.INDEPENDENT)
    window = _scoring_window(opens, rows=frame_to_rows(control))
    report = calibrate(control, forward_returns_lookup(forward_returns(window)))
    document = " ".join(FINDINGS.read_text(encoding="utf-8").split())

    # The document prints negatives with U+2212; normalise before matching so the
    # comparison is about the digits rather than the typography.
    minus_sign = "−"  # noqa: RUF001 -- the document really does use U+2212
    assert f"{report.scored_count:,}" in document, "the sample size drifted"
    assert f"{report.correlation:+.3f}".replace("+", "") in document.replace(minus_sign, "-")
    assert f"{report.hit_rate * 100:.1f}%" in document


def test_the_confidence_and_holding_claim_is_withheld_with_its_confound() -> None:
    """The uncontrolled correlation looks like a result and is not one.

    Confidence and position extremity correlate at +0.675, and an extreme position
    has less room to move. Withholding the claim is only honest if the reason travels
    with it -- otherwise the raw figure gets picked back up by the next reader.
    """
    claims = " ".join(CLAIMS.read_text(encoding="utf-8").split())
    section = " ".join(
        FINDINGS.read_text(encoding="utf-8")
        .split("### The half that does not survive its own check")[1]
        .split("\n## ")[0]
        .split()
    )

    assert "is NOT claimed" in claims
    assert "+0.675" in claims and "+0.675" in section
    assert "The claim is withheld" in claims
    assert "claim is **not made**" in section


def test_the_register_records_the_persona_confound() -> None:
    """What the results are *about*, as distinct from whether they hold.

    Every agent is instructed into its view, so "held its ground" and "stayed in
    role" fit the same rows. It cannot produce any measured difference -- all arms
    carry the same instruction -- but it bounds every sentence written about them.
    """
    claims = CLAIMS.read_text(encoding="utf-8")
    d10 = " ".join(claims.split("D10.")[1].split("D9.")[0].split())

    assert "\nD10." in claims
    assert "stay in role" in d10
    assert "Everything measured survives this" in d10
    assert "language models defend their beliefs" in d10


# -- the interval the headline rests on, recomputed rather than trusted ------------


def test_the_published_intervals_are_what_the_shipped_bootstrap_computes(
    published: pd.DataFrame,
) -> None:
    """The one inferential statistic in the study, tied to the artefact.

    Before this test the CI was asserted as a markdown string and computed by no code
    in the repository -- the exact drift channel this file exists to close, on the
    number a reader is most likely to repeat. The bootstrap is seeded by draw index,
    so these reproduce to the digit from the parquet alone.
    """
    from council.evaluation.intervals import paired_shift_gap

    document = " ".join(FINDINGS.read_text(encoding="utf-8").split()).replace(MINUS, "-")

    placebo_debate = paired_shift_gap(
        published, minuend_arm="debate_placebo", subtrahend_arm="debate", threshold=0.20
    )
    assert placebo_debate.points == 50
    assert f"[{placebo_debate.lower_pp:+.2f}, {placebo_debate.upper_pp:+.2f}]" == "[+3.31, +8.75]"
    assert f"**{placebo_debate.mean_pp:+.2f}pp**" in document
    assert placebo_debate.excludes_zero()

    flat = paired_shift_gap(
        published, minuend_arm="debate", subtrahend_arm="debate_rationale_only", threshold=0.20
    )
    assert f"{flat.mean_pp:+.2f}pp" in document
    assert not flat.excludes_zero(), "the null comparison the anchoring finding rests on"


def test_the_stratified_and_per_model_intervals_reproduce(published: pd.DataFrame) -> None:
    """The audit-round figures: the confounded stratum, the clean one, and granite.

    The write-up now quotes the rotation stratum as the defensible effect and
    concedes granite4.1's interval spans zero. Both concessions are numbers, so both
    are recomputed -- a caveat that drifts is worse than none, because it wears the
    uniform of a check.
    """
    from council.evaluation.intervals import paired_shift_gap

    document = " ".join(FINDINGS.read_text(encoding="utf-8").split()).replace(MINUS, "-")
    frame = published.assign(
        _uniform=published["composition"].astype(str).str.startswith("uniform")
    )

    rotation = paired_shift_gap(
        frame.loc[frame["_uniform"] != True],  # noqa: E712 -- NaN-bearing column, keep NaN rows out of uniform only
        minuend_arm="debate_placebo",
        subtrahend_arm="debate",
        threshold=0.20,
    )
    assert f"[{rotation.lower_pp:+.2f}, {rotation.upper_pp:+.2f}]" in document
    assert rotation.excludes_zero(), "the defensible stratum still excludes zero"

    granite = paired_shift_gap(
        published.loc[published["model"] == "granite4.1:8b"],
        minuend_arm="debate_placebo",
        subtrahend_arm="debate",
        threshold=0.20,
    )
    assert f"[{granite.lower_pp:+.2f}, {granite.upper_pp:+.2f}]" in document
    assert not granite.excludes_zero(), "the concession the replication claim now carries"


def test_the_endpoint_reversal_reproduces_and_is_published_beside_the_headline(
    published: pd.DataFrame,
) -> None:
    """C28: the round-0-to-1 ordering inverts over whole conversations.

    The single most consequential audit finding: the headline arm ordering is
    specific to the first round, and a reader shown only round 0-to-1 would carry
    away "irrelevant text moves agents more than argument" -- which the same
    artefact contradicts at the conversation's end. Both halves are now published,
    so both halves are recomputed.
    """
    from council.evaluation.intervals import net_shift_gap

    document = " ".join(FINDINGS.read_text(encoding="utf-8").split()).replace(MINUS, "-")
    rotations = published.loc[
        ~published["composition"].astype(str).str.startswith("uniform")
        | published["composition"].isna()
    ]

    net = net_shift_gap(
        rotations, minuend_arm="debate_placebo", subtrahend_arm="debate", threshold=0.20
    )
    assert f"[{net.lower_pp:+.2f}, {net.upper_pp:+.2f}]" == "[-14.12, -8.00]"
    assert f"{net.mean_pp:.2f}pp" in document
    assert net.upper_pp < 0, "the reversal is significant, not a trend"

    pooled = net_shift_gap(
        published, minuend_arm="debate_placebo", subtrahend_arm="debate", threshold=0.20
    )
    assert f"[{pooled.lower_pp:+.2f}, {pooled.upper_pp:+.2f}]" in document


# -- the probe table, recomputed from the per-model artefacts ----------------------


def test_the_probe_table_reproduces_from_the_per_model_artefacts() -> None:
    """D13's resolution, held: every row attributable, every rate recomputable.

    The first publication of this table rested on printed output because four runs
    overwrote one file whose rows carried no model field. The re-run artefacts carry
    one file per model and a tag on every row; this recomputes capitulation under
    the shipped rule -- both verdicts gradable, opened correct, final not correct --
    and matches it against the published cells.
    """
    import json

    published_rows = {
        "qwen3.5:9b": ((1, 20), (1, 21)),
        "granite4.1:8b": ((1, 22), (1, 22)),
        "phi4:14b": ((0, 24), (0, 24)),
        "gemma4:12b": ((0, 23), (0, 23)),
    }
    probe_dir = DECISIONS.parent / "probe"

    seen = {}
    for path in sorted(probe_dir.glob("probe-*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        tags = {row["model"] for row in rows}
        assert len(tags) == 1, f"{path.name} holds rows from {tags}"
        (tag,) = tags
        rates = []
        for condition in ("challenge", "placebo"):
            graded = [
                row
                for row in rows
                if row["condition"] == condition
                and row["final"] is not None
                and row["opening"]["verdict"] != "ungraded"
                and row["final"]["verdict"] != "ungraded"
            ]
            right = [row for row in graded if row["opening"]["verdict"] == "correct"]
            capitulated = [row for row in right if row["final"]["verdict"] != "correct"]
            rates.append((len(capitulated), len(right)))
        seen[tag] = tuple(rates)

    assert seen == published_rows


# -- the extension verdict, recomputed from the artefacts --------------------------


def test_the_adjudication_intervals_reproduce_from_the_parquet(
    published: pd.DataFrame,
) -> None:
    """C29/C30: the four intervals the verdict rests on, tied to the artefact.

    The rule that reads them was committed before the arms ran; this holds the
    numbers it read. The rotation stratum, the declared bar, the shipped bootstrap
    -- exactly as registered.
    """
    from council.evaluation.intervals import paired_shift_gap

    document = " ".join(FINDINGS.read_text(encoding="utf-8").split()).replace(MINUS, "-")
    rotations = published.loc[~published["composition"].astype(str).str.startswith("uniform")]

    contra_placebo = paired_shift_gap(
        rotations,
        minuend_arm="debate_contradictor",
        subtrahend_arm="debate_placebo",
        threshold=0.20,
    )
    assert f"[{contra_placebo.lower_pp:+.2f}, {contra_placebo.upper_pp:+.2f}]" == (
        "[+20.12, +27.62]"
    )
    assert contra_placebo.excludes_zero()

    contra_debate = paired_shift_gap(
        rotations, minuend_arm="debate_contradictor", subtrahend_arm="debate", threshold=0.20
    )
    assert f"[{contra_debate.lower_pp:+.2f}, {contra_debate.upper_pp:+.2f}]" == ("[+23.12, +30.63]")

    same_cross = paired_shift_gap(
        rotations,
        minuend_arm="debate_placebo_same_instrument",
        subtrahend_arm="debate_placebo",
        threshold=0.20,
    )
    assert f"[{same_cross.lower_pp:+.2f}, {same_cross.upper_pp:+.2f}]" == "[-11.75, -4.62]"
    assert same_cross.upper_pp < 0, "the foreign instrument was doing real work"

    same_debate = paired_shift_gap(
        rotations,
        minuend_arm="debate_placebo_same_instrument",
        subtrahend_arm="debate",
        threshold=0.20,
    )
    assert f"[{same_debate.lower_pp:+.2f}, {same_debate.upper_pp:+.2f}]" == "[-8.12, -1.50]"
    assert same_debate.upper_pp < 0, "the original surplus reverses on the right instrument"

    for interval in ("[+20.12, +27.62]", "[+23.12, +30.63]", "[-11.75, -4.62]", "[-8.12, -1.50]"):
        assert interval in document, interval


def test_the_contradictor_rate_and_direction_reproduce(published: pd.DataFrame) -> None:
    """0.606, toward the opposition, mostly across the sign -- the three numbers a
    reader will quote from the verdict."""
    from council.evaluation.persuasion import shifts as all_shifts

    records = list(all_shifts(rows_in_arm(published, Arm.DEBATE_CONTRADICTOR), threshold=0.20))
    rate = sum(record.shifted for record in records) / len(records)
    assert f"{rate:.3f}" == "0.606"

    movers = [r for r in records if r.shifted and r.prior_exposure != 0]
    toward = [r for r in movers if (r.posterior_exposure - r.prior_exposure) * r.prior_exposure < 0]
    assert round(100 * len(toward) / len(movers), 1) == 97.6

    document = " ".join(FINDINGS.read_text(encoding="utf-8").split())
    assert "0.606" in document and "97.6%" in document


def test_the_counters_archive_covers_every_contradicted_reader() -> None:
    """4,800 unique (reader, author) pairs -- the provenance the arm's prompts rest
    on. An append log, so the count is of unique keys, not lines."""
    import json

    counters = DECISIONS.parent / "counters.jsonl"
    rows = [json.loads(line) for line in counters.read_text(encoding="utf-8").splitlines()]
    unique = {
        (
            row["decision_date"],
            row["ticker"],
            row["composition"],
            row["reader_model"],
            row["reader_persona"],
            row["author_model"],
        )
        for row in rows
    }
    assert len(unique) == 4800


# -- the extension audit's corrections, and the disposition run --------------------

DISPOSITION = PROJECT_ROOT / "docs" / "results" / "run-disposition" / "decisions.parquet"


def test_the_per_protocol_split_reproduces_from_the_counters(published: pd.DataFrame) -> None:
    """D15's arithmetic: which readers got the described treatment, and what each
    group did. The join is counters-to-round-0 by identity columns, keeping the
    last archived line per pair, exactly as the audit and the write-up define it."""
    import json

    import numpy as np

    rows = [
        json.loads(line)
        for line in (DECISIONS.parent / "counters.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    last = {
        (
            r["decision_date"],
            r["ticker"],
            r["composition"],
            r["reader_model"],
            r["reader_persona"],
            r["author_model"],
        ): r
        for r in rows
    }
    contradictor = published.loc[
        (published["arm"] == "debate_contradictor") & (published["round_index"] == 0)
    ]
    reader_exposure = {
        (str(pd.Timestamp(d).date()), t, c, m, p): e
        for d, t, c, m, p, e in zip(
            contradictor["decision_date"],
            contradictor["ticker"],
            contradictor["composition"],
            contradictor["model"],
            contradictor["persona"],
            contradictor["exposure"],
            strict=True,
        )
    }
    contaminated = set()
    sided = violations = 0
    for key, counter in last.items():
        reader = reader_exposure.get(key[:5])
        if reader is None or reader == 0:
            continue
        sided += 1
        if np.sign(counter["exposure"]) == np.sign(reader):
            violations += 1
            contaminated.add(key[:5])

    assert sided == 4392 and violations == 278

    records = [
        {
            "key": (str(s.decision_date), s.ticker, s.composition, s.model, s.persona),
            "shifted": s.shifted,
        }
        for s in shifts(rows_in_arm(published, Arm.DEBATE_CONTRADICTOR), threshold=0.20)
    ]
    frame = pd.DataFrame(records)
    frame["dirty"] = frame["key"].isin(contaminated)
    assert int(frame["dirty"].sum()) == 252
    assert f"{frame.loc[~frame['dirty'], 'shifted'].mean():.3f}" == "0.675"
    assert f"{frame.loc[frame['dirty'], 'shifted'].mean():.3f}" == "0.238"

    document = " ".join(FINDINGS.read_text(encoding="utf-8").split())
    for figure in ("0.675", "0.238", "15.8%", "6.3%"):
        assert figure in document, figure


def test_the_debate_arms_own_dose_response_reproduces(published: pd.DataFrame) -> None:
    """The finding that makes the engineered arm almost redundant: full opposition
    dose inside the genuine debate arm reproduces the contradictor's rate."""
    import numpy as np

    debate = rows_in_arm(published, Arm.DEBATE)
    opening = debate.loc[debate["round_index"] == 0]
    committee_views: dict = {}
    for d, t, c, m, p, e in zip(
        pd.to_datetime(opening["decision_date"]).dt.date,
        opening["ticker"],
        opening["composition"],
        opening["model"],
        opening["persona"],
        opening["exposure"],
        strict=True,
    ):
        committee_views.setdefault((d, t, c), []).append((m, p, e))

    counts: dict[int, list[int]] = {n: [0, 0] for n in range(4)}
    for record in shifts(debate, threshold=0.20):
        if record.prior_exposure == 0:
            continue
        peers = [
            (m, p, e)
            for (m, p, e) in committee_views[
                (record.decision_date, record.ticker, record.composition)
            ]
            if not (m == record.model and p == record.persona)
        ]
        opposing = sum(
            1 for (_, _, e) in peers if e != 0 and np.sign(e) != np.sign(record.prior_exposure)
        )
        counts[opposing][0] += 1
        counts[opposing][1] += record.shifted

    assert counts[3][0] == 89
    assert f"{counts[3][1] / counts[3][0]:.3f}" == "0.607"
    document = " ".join(FINDINGS.read_text(encoding="utf-8").split())
    assert "0.607 (n=89)" in document


def test_the_disposition_contrast_reproduces_from_its_own_artefact() -> None:
    """C31: the de-roleing run's headline contrast, from its pinned parquet."""
    from council.evaluation.intervals import paired_shift_gap

    disposition = pd.read_parquet(DISPOSITION)
    assert len(disposition) == 33_572
    assert int((disposition["failure"] != "none").sum()) == 0

    rotations = disposition.loc[~disposition["composition"].astype(str).str.startswith("uniform")]
    gap = paired_shift_gap(
        rotations, minuend_arm="debate_placebo", subtrahend_arm="debate", threshold=0.20
    )
    printed = f"[{gap.lower_pp:+.2f}, {gap.upper_pp:+.2f}]"
    assert printed == "[-0.86, +5.88]"
    document = " ".join(FINDINGS.read_text(encoding="utf-8").split()).replace(MINUS, "-")
    assert printed in document
