"""Every tunable, in one place.

Defaults describe the small first run rather than the full grid: four agents, two
tickers, two years. That version answers the headline question and finishes in an
evening, which matters more than covering the whole design on the first attempt.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configuration, from ``COUNCIL_``-prefixed environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="COUNCIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- universe -------------------------------------------------------------

    tickers: tuple[str, ...] = ("AAPL", "XOM")
    """Chosen by a rule stated before any result was seen, not by taste.

    The rule: the largest company by market capitalisation on the start date, in
    each of two sectors that behave differently -- technology and energy. Naming
    the rule is what converts "I picked these because I knew they would work" from
    an accusation into a question with an answer.
    """

    start: date = date(2022, 1, 1)
    end: date = date(2023, 12, 31)

    # -- execution ------------------------------------------------------------

    commission_bps: float = Field(default=5.0, ge=0)
    slippage_bps: float = Field(default=5.0, ge=0)
    """Both on by default. A zero-cost backtest is not a result, and the daily arm
    in particular churns on noise once friction is removed."""

    rebalance_threshold: float = Field(default=0.05, ge=0.0, le=2.0)
    """No trade unless the target moves further than this from the held position."""

    # -- what counts as changing your mind ------------------------------------

    # Declared here, before any debate has run, because choosing this threshold
    # after seeing the data is choosing the answer. A move of this size counts as
    # having shifted; a change of sign counts as a reversal regardless of size.
    shift_threshold: float = Field(default=0.20, gt=0.0, le=2.0)

    # Debate only happens where there is something to debate. On days the agents
    # already agree, a conversation cannot change the committee's decision and
    # teaches nothing -- what skipping them saves has never been measured at the
    # committee level; on the pooled grid the contested share was 100%,
    # so it saved nothing.
    dispersion_threshold: float = Field(default=0.25, ge=0.0)

    # -- when a conversation ends ---------------------------------------------
    #
    # The protocol can end a conversation on agreement, on stillness or at the
    # cap, and *which* of those ended it would be a measurement. At the shipped
    # cap of one rebuttal round it is not one: only AGREED and CAP are reachable,
    # and the reason is returned on the transcript rather than stored. See the
    # cap's own comment below, and task 19.
    #
    # These three bounds were *not* declared before the run, and neither was
    # `placebo_min_gap_sessions` below. `shift_threshold` and
    # `dispersion_threshold` above were, in the first commit (afce0ae), before any
    # result existed. These four were added in cbf6a55, after the results committed
    # in fa436fa and 98a4020 -- and `agreement_spread` and
    # `placebo_min_gap_sessions` were calibrated from that run's measured spreads
    # and donor distances rather than chosen blind. They are declared here so a
    # reader can check a result against them; they are not pre-registration and
    # must not be read as such.

    # Agreement: the widest gap between any two seats. Set from the first run's
    # measured spreads rather than by taste. The shares below do **not** reproduce
    # from `docs/results/superseded/run-2models/decisions.parquet`, which gives
    # 39.5% at round 0, 37.3% after one round, 64.5% at a 0.50 bar, 34.1% under a
    # bare `<=`, and 95 committees exactly on a 0.20 spread of which a bare
    # comparison admits 35. They come from the four-model run, whose artefacts are
    # gone. Only 15.6% of committees already satisfy it before saying a word, so
    # nearly every committee genuinely debates; 30.8% reach it after one round, so
    # it is achievable. A looser bar of 0.50 would have stopped 36.6% of committees
    # before they spoke.
    #
    # All three shares are measured with `evaluation.threshold.within`, the same
    # comparison `debate.protocol._agreed` applies, so the number quoted here and
    # the predicate that ships cannot drift apart. They did once: on the file above
    # a bare `<=` reads 34.1% at round 0 against `within`'s 39.5%, because 95
    # committees sit exactly on a 0.20 spread there and binary subtraction admits
    # only 35 of them.
    #
    # It is also the same number as `shift_threshold`, and that is the point: a
    # spread under it means no two agents differ by more than what this study
    # calls changing your mind. They are inside each other's noise.
    agreement_spread: float = Field(default=0.20, gt=0.0, le=2.0)

    # Stillness: consecutive rounds in which no seat moved at all. Two rather than
    # one deliberately -- an agent that ignored an argument on first reading may
    # take it on the second, and stopping at the first quiet round would record
    # that committee as entrenched without giving it the chance.
    #
    # Read by no run at the shipped cap *with this default of two*. A streak of two
    # quiet rounds needs at least two rebuttal rounds, and the cap below is one, so
    # `debate.protocol.StopReason.SETTLED` cannot occur at the shipped cap with
    # `stillness_rounds = 2`. It is reachable at 1 -- a value this field permits,
    # the sweep threads through to `run_debate`, and the test suite parametrises --
    # where one quiet rebuttal round ends the conversation as SETTLED. The bound is
    # kept declared rather than deleted because raising the cap is a change to the
    # experiment, not a tuning knob.
    stillness_rounds: int = Field(default=2, ge=1, le=10)

    # The cap, pinned to one rebuttal round -- the value of
    # `debate.protocol.DEFAULT_REBUTTAL_ROUNDS`, which cannot be imported here
    # without closing an import cycle. Pinned rather than chosen: at any higher
    # value `evaluation.persuasion._by_round` raises on the middle rounds it
    # cannot pair, `scoring.arm_exposures` reads the treatment arm at one fixed
    # round index, and `debate.sweep._Sweep.group`'s resume check demands a row
    # for every round up to the cap. `debate.sweep.run_debate_arms` refuses any
    # other value rather than letting it corrupt a run silently, so raising this
    # fails loudly and names what has to change first.
    max_debate_rounds: int = Field(default=1, ge=1, le=20)

    # How far back the placebo donor must come from, in trading sessions. Added in
    # cbf6a55, after the results in fa436fa and 98a4020, and calibrated from that
    # run's donor distances -- so it is not pre-registered either. The
    # figures below came from the two-model six-month run, which is superseded --
    # its decisions survive at `docs/results/superseded/run-2models/`; they are
    # provenance, not current measurements. The
    # first run drew donors a median of 14 sessions back while agents look 60
    # sessions back -- so a "different day" shared roughly 46 of its 60 bars with
    # the day being decided, and the arm that was supposed to be inert was showing
    # arguments about nearly the same data. At or above `lookback_days` the two
    # windows cannot overlap at all.
    placebo_min_gap_sessions: int = Field(default=60, ge=0)

    # -- generation -----------------------------------------------------------

    ollama_base_url: str = "http://localhost:11434"

    agent_models: tuple[str, ...] = (
        "qwen3.5:9b",
        "granite4.1:8b",
        "phi4:14b",
        "gemma4:12b",
    )
    """Distinct families rather than sizes of one family: architecture is a factor
    in this experiment, and four checkpoints of the same lineage would confound it
    with scale."""

    @field_validator("agent_models")
    @classmethod
    def _models_are_distinct(cls, models: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(models)) != len(models):
            raise ValueError(
                f"duplicate model in {models}; each model takes one seat, and a repeat "
                "runs the whole independent grid twice while debate.compositions "
                "refuses the same configuration"
            )
        return models

    seed: int = 20260101
    temperature: float = 0.0
    max_output_tokens: int = Field(default=320, ge=64)
    context_tokens: int = Field(default=4096, ge=1024)
    keep_alive: str = "30m"
    request_timeout_seconds: float = Field(default=120.0, gt=0)

    concurrency: int = Field(default=4, ge=1, le=32)
    """A single GPU is memory-bandwidth bound; past roughly four concurrent
    requests the queue grows and latency with it."""

    max_retries: int = Field(default=2, ge=0, le=5)

    # Which physical GPU generation may use. The development machine runs other
    # work, so this is a first-class setting rather than an environment detail:
    # pinning here keeps a long run off a card that is already busy.
    cuda_visible_devices: str | None = None

    # -- history ---------------------------------------------------------------

    lookback_days: int = Field(default=60, ge=5)
    """How much price history an agent sees. Long enough for a reversion reader to
    have something to revert to."""

    # -- paths ----------------------------------------------------------------

    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def prices_path(self) -> Path:
        return self.data_dir / "prices.parquet"

    @property
    def decisions_path(self) -> Path:
        return self.data_dir / "decisions.parquet"

    @property
    def completions_path(self) -> Path:
        """Full prompts and raw completions, appended as JSON lines.

        Kept in full. A new aggregation rule, an anonymisation audit or a
        rationale analysis can then be run later at zero additional inference
        cost, which is the difference between a follow-up question taking an hour
        and taking another overnight run.
        """
        return self.data_dir / "completions.jsonl"

    @property
    def total_cost_bps(self) -> float:
        return self.commission_bps + self.slippage_bps


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings instance."""
    return Settings()
