"""Finding a finished run, and saying what is missing when there is not one.

The dashboard reads artefacts. It calls no model, regenerates nothing, and holds
no number that is not already on disk -- so the only two inputs are the
consolidated decisions frame and the price history it was scored against.

When either is absent the honest output is a sentence naming the path and what
writes it. A dashboard that renders empty axes instead invites the reader to
believe a run happened and produced nothing, which is a different and much worse
claim than "no run has happened yet".
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from council.data.prices import load_prices, opens_frame
from council.domain.signal import Arm
from council.evaluation.frames import (
    ARM,
    COMPOSITION,
    NO_COMPOSITION,
    REQUIRED_COLUMNS,
    TICKER,
)

RATIONALE: Final = "rationale"
"""The one stored column the evaluation package never reads and this app does.

Every statistic in :mod:`council.evaluation` is computed from exposures and
confidences. The transcript panel needs what the agents actually said, which is
stored but is not part of :data:`~council.evaluation.frames.REQUIRED_COLUMNS`.
"""

DASHBOARD_COLUMNS: Final[tuple[str, ...]] = (*REQUIRED_COLUMNS, RATIONALE)

ARM_ORDER: Final[tuple[str, ...]] = tuple(str(arm) for arm in Arm)
"""Arms in the order :class:`~council.domain.signal.Arm` declares them.

Taken from the enum rather than written out, so that a panel cannot order the
arms differently from another panel, and so that the debate arm and its placebo
always sit next to each other -- the gap between those two is the finding, and a
reader should not have to hunt across a table for it.
"""

DECISIONS_SOURCE: Final = (
    "`python -m council generate` runs the independent arm and "
    "`python -m council debate` runs the debate arms over the contested points. "
    "Each writes one part file per (model, persona, ticker), and "
    "council.agents.store.DecisionStore.consolidate folds the parts into this "
    "single frame."
)

PRICES_SOURCE: Final = (
    "A long-format price parquet with the columns council.data.prices.validate_prices "
    "requires. `python -m council dryrun --synthetic` writes a generated one beside "
    "the decisions it produces, which is the offline path; a real run points "
    "COUNCIL_DATA_DIR at a directory holding a downloaded history under this name."
)


def order_arms(arms: Iterable[str]) -> tuple[str, ...]:
    """Known arms in the declared order, then anything unrecognised, sorted.

    An unknown arm is kept rather than dropped: a frame carrying one is a frame
    this code does not understand, and hiding it would present a partial table as
    a complete one.
    """
    present = set(arms)
    known = tuple(arm for arm in ARM_ORDER if arm in present)
    return (*known, *sorted(present.difference(ARM_ORDER)))


@dataclass(frozen=True, slots=True)
class MissingArtefact:
    """One file the dashboard needs, and what produces it."""

    label: str
    path: Path
    produced_by: str


@dataclass(frozen=True, slots=True)
class ArtefactStatus:
    """What is on disk, reported as what is not."""

    missing: tuple[MissingArtefact, ...]

    @property
    def is_ready(self) -> bool:
        return not self.missing


class MissingArtefactsError(FileNotFoundError):
    """Raised instead of returning an empty frame.

    An empty frame would flow through every panel and come out as a flat equity
    curve and a shift rate of zero -- numbers that look like results.
    """

    def __init__(self, status: ArtefactStatus) -> None:
        self.status = status
        super().__init__(
            "no run to read: " + ", ".join(str(item.path) for item in status.missing)
        )


def artefact_status(*, decisions_path: Path, prices_path: Path) -> ArtefactStatus:
    """Which of the two required files are absent, in the order a reader needs them."""
    candidates = (
        MissingArtefact(label="decisions", path=decisions_path, produced_by=DECISIONS_SOURCE),
        MissingArtefact(label="prices", path=prices_path, produced_by=PRICES_SOURCE),
    )
    return ArtefactStatus(
        missing=tuple(item for item in candidates if not item.path.is_file())
    )


def require_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject a decisions frame the panels cannot read.

    Raises:
        ValueError: naming every absent column at once. A frame written by an
            older version of the store is missing a column, not a run, and the
            reader needs to be told which rather than shown a KeyError from
            whichever panel happens to be drawn first.
    """
    missing = [column for column in DASHBOARD_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"decisions frame is missing columns: {', '.join(missing)}")
    return frame


@dataclass(frozen=True, slots=True)
class Results:
    """One run: every stored decision, and the opens it is scored against."""

    decisions: pd.DataFrame
    opens: pd.DataFrame

    @property
    def is_empty(self) -> bool:
        """Whether the file exists but holds nothing.

        Distinct from the file being absent, and worth distinguishing: an empty
        parquet means a run started and stored no decision, which is a fault
        rather than a state to wait out.
        """
        return self.decisions.empty

    @property
    def arms(self) -> tuple[str, ...]:
        return order_arms(str(value) for value in self.decisions[ARM])

    @property
    def compositions(self) -> tuple[str, ...]:
        """The committees present, excluding the independent arm's null composition."""
        present = {str(value) for value in self.decisions[COMPOSITION].fillna(NO_COMPOSITION)}
        return tuple(sorted(present - {NO_COMPOSITION}))

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(sorted({str(value) for value in self.decisions[TICKER]}))

    def scoped_to(self, composition: str | None) -> Results:
        """This run narrowed to one committee, keeping the control it is read against.

        ``None`` returns the run unchanged, which is the declared scope: the
        balanced design is one experiment rather than eight.

        The independent arm carries no composition, so filtering the column alone
        would delete the control from every panel that shows a committee beside
        it -- the calibration panel would lose its baseline population and the
        shift panel the arm that cannot shift by construction. Both are kept.
        """
        if composition is None:
            return self
        held = self.decisions[COMPOSITION].fillna(NO_COMPOSITION).astype(str)
        return Results(
            decisions=self.decisions.loc[held.isin({composition, NO_COMPOSITION})],
            opens=self.opens,
        )


def load_results(*, decisions_path: Path, prices_path: Path) -> Results:
    """Read both artefacts, or say which one is not there.

    The price panel is narrowed to the tickers the decisions actually cover. The
    buy-and-hold curve is an equal-weight basket of whatever columns it is handed,
    so leaving an untraded ticker in would compare the committee against a
    different universe from the one it was asked about.

    Raises:
        MissingArtefactsError: if either file is absent.
        ValueError: if the decisions frame is missing a column, or a decision
            names a ticker the price file has no history for.
    """
    status = artefact_status(decisions_path=decisions_path, prices_path=prices_path)
    if not status.is_ready:
        raise MissingArtefactsError(status)

    decisions = require_columns(pd.read_parquet(decisions_path))
    covered = sorted({str(value) for value in decisions[TICKER]})
    prices = load_prices(prices_path)
    return Results(decisions=decisions, opens=opens_frame(prices, tickers=covered or None))
