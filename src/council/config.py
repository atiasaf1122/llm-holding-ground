"""Every tunable, in one place.

Defaults describe the small first run rather than the full grid: four agents, two
tickers, two years. That version answers the headline question and finishes in an
evening, which matters more than covering the whole design on the first attempt.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field
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
    # teaches nothing -- and skipping them is most of the compute budget.
    dispersion_threshold: float = Field(default=0.25, ge=0.0)

    # -- generation -----------------------------------------------------------

    ollama_base_url: str = "http://localhost:11434"

    agent_models: tuple[str, ...] = ("qwen3:8b", "gemma4:latest")
    """Distinct families rather than sizes of one family: architecture is a factor
    in this experiment, and four checkpoints of the same lineage would confound it
    with scale."""

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
