"""Whether the command line does what it says and exits with a code that means it.

Every command here runs with ``--mock`` and ``--synthetic``, so the suite needs no
daemon and no price file. The exit codes are asserted as carefully as the output:
a wrapper script that treats a failed night as success is how an outage gets
published as a result.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from council.agents.mock import MockProvider
from council.agents.provider import ProviderUnavailableError
from council.agents.runner import CUDA_DEVICES_VAR
from council.app.artefacts import artefact_status
from council.cli import EXIT_FAILURE, EXIT_OK, main
from council.config import get_settings
from council.data.prices import load_prices, synthetic_prices, write_prices
from helpers_pipeline import MODELS, START, TICKERS

END = date(2022, 2, 4)
"""A month of sessions: enough to clear the five-session warm-up these tests pin
and leave real decisions behind it, short enough that the whole CLI suite runs in
seconds on the mock provider."""


GENERATING = ("generate", "debate", "probe")
"""The subcommands that call a model, and so the ones that take ``--mock``."""


def command(*extra: str, data_dir: Path) -> list[str]:
    """One subcommand's arguments, pinned so no test depends on the environment."""
    return [
        *extra,
        "--tickers",
        *TICKERS,
        "--models",
        *MODELS,
        "--start",
        START.isoformat(),
        "--end",
        END.isoformat(),
        "--data-dir",
        str(data_dir),
        "--synthetic",
        "--log-level",
        "ERROR",
        *(["--mock"] if extra and extra[0] in GENERATING else []),
    ]


def run(*extra: str, data_dir: Path) -> tuple[int, str]:
    out = io.StringIO()
    code = main(command(*extra, data_dir=data_dir), out=out)
    return code, out.getvalue()


@pytest.fixture(autouse=True)
def _short_lookback(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Five sessions of warm-up, so a month of prices still yields decisions.

    Through the environment and the settings cache, because that is the path a
    developer's own ``.env`` takes; a monkeypatched attribute would test a
    configuration no run can actually be given.
    """
    monkeypatch.setenv("COUNCIL_LOOKBACK_DAYS", "5")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# -- plan -------------------------------------------------------------------------


def test_plan_prints_a_table_and_generates_nothing(tmp_path: Path) -> None:
    code, printed = run("plan", data_dir=tmp_path)

    assert code == EXIT_OK
    assert "inferences" in printed
    assert "total" in printed
    assert not (tmp_path / "decisions.parquet").exists()


def test_plan_reports_a_measured_share_once_the_control_arm_exists(tmp_path: Path) -> None:
    run("generate", data_dir=tmp_path)

    _, printed = run("plan", data_dir=tmp_path)

    assert "contested (measured)" in printed
    assert "* estimated" not in printed


# -- generate, debate, evaluate ---------------------------------------------------


def test_generate_then_debate_then_evaluate_writes_a_results_file(tmp_path: Path) -> None:
    assert run("generate", data_dir=tmp_path)[0] == EXIT_OK
    assert run("debate", data_dir=tmp_path)[0] == EXIT_OK

    code, printed = run("evaluate", data_dir=tmp_path)

    assert code == EXIT_OK
    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert [arm["arm"] for arm in payload["arms"]] == [
        "independent",
        "debate",
        "debate_rationale_only",
        "debate_placebo",
    ]
    assert "Shift rate by prior confidence" in printed


def test_a_second_generate_reports_everything_already_stored(tmp_path: Path) -> None:
    run("generate", data_dir=tmp_path)

    code, printed = run("generate", data_dir=tmp_path)

    assert code == EXIT_OK
    assert "generated 0" in printed


def test_a_night_in_which_every_generation_failed_does_not_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unreachable daemon writes a flat row for every point. Exiting 0 on that
    # would let a wrapper move on to the debate arms over a control arm that is
    # flat everywhere, and the outage would be published as a finding.
    monkeypatch.setattr(
        "council.cli.mock_factory",
        lambda model: MockProvider(model=model, responses=[ProviderUnavailableError("down")]),
    )

    code, _ = run("generate", data_dir=tmp_path)

    assert code == EXIT_FAILURE


def test_the_requested_device_is_pinned_before_any_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CUDA_DEVICES_VAR, raising=False)

    code, _ = run("generate", "--device", "1", data_dir=tmp_path)

    assert code == EXIT_OK
    assert os.environ[CUDA_DEVICES_VAR] == "1"


def test_debating_before_generating_fails_rather_than_reporting_success(
    tmp_path: Path,
) -> None:
    code, _ = run("debate", data_dir=tmp_path)

    assert code == EXIT_FAILURE


def test_evaluating_before_generating_fails_rather_than_reporting_success(
    tmp_path: Path,
) -> None:
    code, _ = run("evaluate", data_dir=tmp_path)

    assert code == EXIT_FAILURE


def test_a_missing_price_file_fails_rather_than_backtesting_nothing(tmp_path: Path) -> None:
    # Without --synthetic there is no parquet in a fresh directory. An empty frame
    # here would become an empty backtest reporting a return of zero.
    code = main(["plan", "--data-dir", str(tmp_path), "--log-level", "ERROR"])

    assert code == EXIT_FAILURE


def test_an_unknown_subcommand_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["nonesuch"])

    assert exited.value.code == 2


def test_an_unknown_log_level_is_a_usage_error_rather_than_a_traceback(tmp_path: Path) -> None:
    # logging.basicConfig runs before the try block in main, so an unconstrained
    # level would raise ValueError("Unknown level") past the one clause that exists
    # to turn an expected failure into a sentence.
    with pytest.raises(SystemExit) as exited:
        main(["plan", "--synthetic", "--data-dir", str(tmp_path), "--log-level", "bogus"])

    assert exited.value.code == 2


def test_a_lower_case_log_level_is_still_accepted(tmp_path: Path) -> None:
    code = main(["plan", "--synthetic", "--data-dir", str(tmp_path), "--log-level", "error"])

    assert code == EXIT_OK


def test_the_published_results_file_is_valid_json_and_not_merely_python_json(
    tmp_path: Path,
) -> None:
    # `Infinity` is what json.dump writes for a non-finite float by default, and it
    # is not a JSON token: jq, JSON.parse and every schema validator reject it, so
    # the artefact would be readable only by the language that wrote it.
    def reject(token: str) -> object:
        raise ValueError(f"not valid JSON: the token {token!r}")

    run("generate", data_dir=tmp_path)
    run("debate", data_dir=tmp_path)
    run("evaluate", data_dir=tmp_path)

    text = (tmp_path / "results.json").read_text(encoding="utf-8")

    assert json.loads(text, parse_constant=reject)


def test_an_exploratory_aggregation_says_so_and_the_declared_one_does_not(
    tmp_path: Path,
) -> None:
    run("generate", data_dir=tmp_path)

    _, declared = run("evaluate", data_dir=tmp_path)
    _, exploratory = run("evaluate", "--rule", "median", data_dir=tmp_path)

    assert "* exploratory" not in declared
    assert "* exploratory" in exploratory


# -- dryrun -----------------------------------------------------------------------


def test_dryrun_runs_every_stage_and_writes_its_results_somewhere_of_its_own(
    tmp_path: Path,
) -> None:
    code, printed = run("dryrun", data_dir=tmp_path)

    assert code == EXIT_OK
    assert "wall clock" in printed  # the plan stage, which nothing downstream reads
    assert "contested point(s) to debate" in printed
    assert "Net influence" in printed
    assert (tmp_path / "results.json").is_file()


def test_a_dry_run_leaves_artefacts_the_dashboard_can_read(tmp_path: Path) -> None:
    # The dry run is the documented offline rehearsal, and the dashboard reads its
    # prices off disk. Synthesised in memory only, the rehearsal would produce a
    # decisions parquet with nothing to score it against.
    run("dryrun", data_dir=tmp_path)

    status = artefact_status(
        decisions_path=tmp_path / "decisions.parquet", prices_path=tmp_path / "prices.parquet"
    )

    assert status.is_ready
    assert not load_prices(tmp_path / "prices.parquet").empty


def test_a_synthetic_run_never_overwrites_a_price_file_already_on_disk(tmp_path: Path) -> None:
    # `--synthetic` says which prices this run uses, not which downloaded history
    # to replace; a night's download is expensive and generated prices are not.
    downloaded = synthetic_prices(tickers=TICKERS, start=START, sessions=40, seed=99)
    write_prices(downloaded, tmp_path / "prices.parquet")
    before = (tmp_path / "prices.parquet").read_bytes()

    run("generate", data_dir=tmp_path)

    assert (tmp_path / "prices.parquet").read_bytes() == before


def test_dryrun_keeps_its_decisions_out_of_the_configured_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A mock decision and a real one are the same shape on disk, so a dry run into
    # `data/` would satisfy the resume check for a night of real generation.
    monkeypatch.setenv("COUNCIL_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()

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
            END.isoformat(),
            "--log-level",
            "ERROR",
        ]
    )

    assert code == EXIT_OK
    assert (tmp_path / "dryrun" / "decisions.parquet").is_file()
    assert not (tmp_path / "decisions.parquet").exists()


# -- probe --------------------------------------------------------------------------


def test_probe_scores_the_corpus_and_leaves_its_trials_on_disk(tmp_path: Path) -> None:
    # The probe had no entry point at all: it could not be launched, and nothing it
    # produced was written down, so a run could not be audited or re-scored.
    code, printed = run("probe", data_dir=tmp_path)

    assert code == EXIT_OK
    assert "capitulation probe on" in printed
    assert "prior confidence" in printed
    assert (tmp_path / "probe.jsonl").is_file()


def test_probe_archives_one_readable_row_per_trial(tmp_path: Path) -> None:
    run("probe", "--out", str(tmp_path / "trials.jsonl"), data_dir=tmp_path)

    rows = [
        json.loads(line)
        for line in (tmp_path / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert {row["condition"] for row in rows} == {"challenge", "placebo"}
    assert all(row["opening"]["prompt_hash"] for row in rows)


def test_probe_runs_the_model_it_was_asked_for(tmp_path: Path) -> None:
    _, printed = run("probe", "--model", "some-tag:8b", data_dir=tmp_path)

    assert "some-tag:8b" in printed


def test_a_probe_in_which_every_generation_failed_does_not_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "council.cli.mock_factory",
        lambda model: MockProvider(model=model, responses=[ProviderUnavailableError("down")]),
    )

    code, _ = run("probe", data_dir=tmp_path)

    assert code == EXIT_FAILURE
