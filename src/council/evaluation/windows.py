"""Did the treatment beat the control, or did it beat it once?

A single equity curve over two years is one observation. Cutting the period into a
handful of non-overlapping windows and counting how many the treatment won is a
cruder measure and a more honest one: a strategy that wins because of three days in
March shows up here as one window out of five.

What this deliberately does not report is a p-value. With five windows there is no
test worth running, and attaching a number with three decimal places to it would
dress a count of wins up as an inference it cannot support. The count is the
result; the reader can see how few windows it came from.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

# Lazily evaluated: ``pd.Series`` is only subscriptable to a type checker.
type ReturnSeries = Sequence[float] | npt.NDArray[np.float64] | pd.Series[float]


@dataclass(frozen=True, slots=True)
class Window:
    """One period, and what each arm earned over it."""

    index: int
    start: int
    stop: int
    """Half-open, so ``stop`` is the first period *not* in this window."""

    treatment_return: float
    control_return: float

    @property
    def length(self) -> int:
        return self.stop - self.start

    @property
    def margin(self) -> float:
        return self.treatment_return - self.control_return

    @property
    def treatment_won(self) -> bool:
        """Strictly greater. A dead heat is not a win."""
        return self.treatment_return > self.control_return

    @property
    def is_tie(self) -> bool:
        return self.treatment_return == self.control_return


@dataclass(frozen=True, slots=True)
class WindowComparison:
    """The count, and the windows it was counted from."""

    windows: tuple[Window, ...]

    @property
    def window_count(self) -> int:
        return len(self.windows)

    @property
    def treatment_wins(self) -> int:
        return sum(1 for window in self.windows if window.treatment_won)

    @property
    def ties(self) -> int:
        """Kept separate from losses. Identical arms tie every window, and a run
        reporting zero wins should be distinguishable from one reporting no
        difference at all."""
        return sum(1 for window in self.windows if window.is_tie)

    @property
    def summary(self) -> str:
        return f"{self.treatment_wins} of {self.window_count} windows"


def split_windows(length: int, window_count: int) -> tuple[tuple[int, int], ...]:
    """Cut ``length`` periods into ``window_count`` contiguous, non-overlapping spans.

    Every period lands in exactly one window. Where the split is uneven the earlier
    windows take the extra period each, rather than the remainder being dropped: a
    trailing stub silently discarded is the tail of the sample, and the tail is
    where a strategy that has stopped working shows it.

    Raises:
        ValueError: if there are fewer periods than windows, or fewer than one
            window. Both would produce empty windows whose return is undefined.
    """
    if window_count < 1:
        raise ValueError("need at least one window")
    if length < window_count:
        raise ValueError(f"cannot cut {length} periods into {window_count} non-empty windows")

    base, remainder = divmod(length, window_count)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(window_count):
        stop = start + base + (1 if index < remainder else 0)
        bounds.append((start, stop))
        start = stop
    return tuple(bounds)


def compound(returns: npt.NDArray[np.float64]) -> float:
    """Total return over a span of period returns.

    Compounded rather than summed, because that is what the capital did.
    """
    return float(np.prod(1.0 + returns) - 1.0)


def compare_windows(
    treatment: ReturnSeries, control: ReturnSeries, *, window_count: int
) -> WindowComparison:
    """Score both arms window by window and count the treatment's wins.

    Args:
        treatment: per-period net returns of the debate arm.
        control: per-period net returns of the independent arm, over the same
            periods in the same order.

    Raises:
        ValueError: if either arm is not one flat series of per-period returns, or if
            the two differ in length. They must come from the same backtest calendar;
            comparing misaligned windows would compare different months to each other
            and report the difference as an effect.
    """
    treatment_returns = np.asarray(treatment, dtype=np.float64)
    control_returns = np.asarray(control, dtype=np.float64)
    # Shape before length, because the length message indexes into the shape: a
    # scalar arm checked the other way round raises IndexError from inside the
    # message that was meant to explain the problem.
    if treatment_returns.ndim != 1 or control_returns.ndim != 1:
        raise ValueError("expected one return per period, as a flat series")
    if treatment_returns.shape != control_returns.shape:
        raise ValueError(
            f"arms must cover the same periods; got {treatment_returns.shape[0]} "
            f"and {control_returns.shape[0]}"
        )

    windows = tuple(
        Window(
            index=index,
            start=start,
            stop=stop,
            treatment_return=compound(treatment_returns[start:stop]),
            control_return=compound(control_returns[start:stop]),
        )
        for index, (start, stop) in enumerate(
            split_windows(int(treatment_returns.shape[0]), window_count)
        )
    )
    return WindowComparison(windows=windows)
