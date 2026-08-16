"""``python -m council`` -- the six things a developer actually does.

``plan`` costs a configuration without running it, ``generate`` sweeps the
independent arm, ``debate`` runs the three treatment arms over the contested
points, ``evaluate`` scores what is on disk, and ``dryrun`` does all four in that
order on synthetic prices and the mock provider with no GPU at all. ``probe`` is
the odd one out: it runs the capitulation probe, which shares the provider and the
prompt conventions but touches no prices, no dates and no tickers.

They are separate subcommands rather than flags on one because they are separated
in time. Generation is an overnight job; evaluation is a question asked repeatedly
the next morning. Each resumes from :class:`~council.agents.store.DecisionStore`,
so running ``generate`` twice costs nothing and interrupting it costs one
checkpoint.

Exit codes are load-bearing: 0 only when the command did what it said. A run whose
every generation failed exits 1, because a wrapper script that treats a night of
unreachable-daemon rows as success is how an outage gets published as a result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final, TextIO

import pandas as pd

from council.agents.mock import MockProvider
from council.agents.provider import Provider, ProviderError
from council.agents.runner import (
    GenerationRunner,
    ProviderFactory,
    ollama_factory,
    pin_device,
)
from council.agents.store import DecisionStore
from council.config import Settings, get_settings
from council.data.prices import write_prices
from council.debate.sweep import run_debate_arms
from council.evaluation.aggregation import RULES
from council.pipeline import (
    generate_independent,
    load_or_synthesise_prices,
    open_store,
    select_contested,
    stored_decisions,
)
from council.planning import SECONDS_PER_INFERENCE, TREATMENT_ARMS, plan_experiment
from council.probe.render import render_probe
from council.probe.session import probe_model
from council.report import render_plan, render_results, results_as_json
from council.scoring import DEFAULT_WINDOW_COUNT, PRIMARY_RULE, evaluate_experiment

EXIT_OK: Final = 0
EXIT_FAILURE: Final = 1

RESULTS_FILENAME: Final = "results.json"

LOG_LEVELS: Final[tuple[str, ...]] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

DRYRUN_SUBDIR: Final = "dryrun"
"""Where a dry run writes. Its own directory, because a mock decision and a real
one are the same shape on disk, and a resumable store cannot tell them apart --
one dry run into ``data/`` would silently satisfy the resume check for a night of
real generation that then never happens."""

DRYRUN_SESSIONS: Final = 90
"""Business days a dry run covers unless told otherwise. Long enough to clear the
warm-up and leave real decisions behind it, short enough to finish in seconds."""

_LOG = logging.getLogger("council")


def window_count(value: str) -> int:
    """``--windows``, refused below one rather than silently clamped up to it.

    :func:`council.scoring._arm_reports` clamps the count into the periods the run
    actually holds, which is what lets a short dry run report anything at all. That
    clamp also turned ``--windows 0`` and ``--windows -5`` into one window with no
    complaint, and the report then printed "x of 1 windows" as though one had been
    asked for. A count below one is not a short run, it is a typo, so it is refused
    at the boundary; the downward clamp stays where it is and now logs.
    """
    try:
        count = int(value)
    except ValueError as not_an_int:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from not_an_int
    if count < 1:
        raise argparse.ArgumentTypeError(f"--windows must be at least 1, not {count}")
    return count


# -- settings ---------------------------------------------------------------------


def settings_from(args: argparse.Namespace) -> Settings:
    """Apply the command-line overrides to the configured settings.

    Rebuilt through the model rather than copied into it, so an override is
    validated the same way an environment variable is: a duplicated ticker or a
    negative lookback is refused here rather than three steps into a sweep.
    """
    overrides: dict[str, Any] = {}
    if args.tickers:
        overrides["tickers"] = tuple(args.tickers)
    if args.models:
        overrides["agent_models"] = tuple(args.models)
    if args.start is not None:
        overrides["start"] = args.start
    if args.end is not None:
        overrides["end"] = args.end
    if args.device is not None:
        overrides["cuda_visible_devices"] = args.device
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if getattr(args, "placebo_min_gap", None) is not None:
        overrides["placebo_min_gap_sessions"] = args.placebo_min_gap
    return Settings(**{**get_settings().model_dump(), **overrides})


def dryrun_settings(args: argparse.Namespace) -> Settings:
    """The dry run's own configuration: a short calendar and a directory of its own.

    Both are defaults rather than impositions -- an explicit ``--end`` or
    ``--data-dir`` wins -- so the same command can be pointed at the full range
    when the point is to time it.
    """
    settings = settings_from(args)
    updates: dict[str, Any] = {}
    if args.end is None:
        calendar = pd.bdate_range(start=settings.start, periods=DRYRUN_SESSIONS)
        updates["end"] = calendar[-1].date()
    if args.data_dir is None:
        updates["data_dir"] = settings.data_dir / DRYRUN_SUBDIR

    # A dry run is a few dozen sessions, and the production placebo gap is a full
    # lookback window -- no donor could satisfy it here, and every placebo
    # conversation would be abandoned. Waived, not weakened: a gap of 1 admits the
    # session immediately before the decision, which is arithmetically the same set
    # of donors a gap of 0 admits, so the dry run's placebo carries no
    # donor-distance control at all. Said out loud rather than left for a reader to
    # infer from a suspiciously clean run: the placebo in a dry run is not the
    # control the real arm gets, and a dry run's numbers were never results in the
    # first place.
    if settings.placebo_min_gap_sessions > 1:
        # Unconditionally, and not on a calculation about the calendar. The pool a
        # donor is drawn from is the *independent* arm's sessions -- every one of
        # them, see `debate.sweep.placebo_pool_for` -- and after the
        # `lookback_days - 1` warm-up a 90-session dry run leaves about 31 decision
        # dates to fill it, fewer than the configured 60 the gap asks for. That
        # margin moves with `--start`, `--end` and `lookback_days`, so a dry run
        # that recomputed it and guessed wrong would abandon every placebo
        # conversation and still exit zero.
        updates["placebo_min_gap_sessions"] = 1
        _LOG.warning(
            "dry run: placebo donor gap reduced from %d sessions to 1. Its calendar "
            "cannot support the configured gap, so the placebo here is not the "
            "control the real arm gets. Smoke test only, not a result.",
            settings.placebo_min_gap_sessions,
        )
    return settings.model_copy(update=updates)


def mock_factory(model: str) -> Provider:
    """A provider that answers from a digest of the prompt. No daemon, no GPU."""
    return MockProvider(model=model)


def _factory(settings: Settings, *, mock: bool) -> ProviderFactory:
    return mock_factory if mock else ollama_factory(settings)


# -- subcommands ------------------------------------------------------------------


def do_plan(args: argparse.Namespace, out: TextIO) -> int:
    """Count the inferences a configuration implies. Issues none of them."""
    settings = settings_from(args)
    # No persistence: a command that promises to issue nothing should not leave a
    # price file behind either.
    prices = load_or_synthesise_prices(settings, synthetic=args.synthetic)
    store = open_store(settings)
    decisions = stored_decisions(store)
    contested = (
        tuple(point.point for point in select_contested(decisions, settings=settings))
        if not decisions.empty
        else None
    )
    plan = plan_experiment(
        settings=settings,
        prices=prices,
        store=store,
        contested=contested,
        # So the placebo stage counts the points it can draw a donor for rather
        # than every contested point, which the sweep would never spend.
        decisions=decisions,
        seconds_per_inference=args.seconds_per_inference,
    )
    print(render_plan(plan), file=out)
    return EXIT_OK


def do_generate(args: argparse.Namespace, out: TextIO) -> int:
    """Run the independent arm, resuming whatever is already stored."""
    settings = settings_from(args)
    pin_device(settings.cuda_visible_devices)
    prices = load_or_synthesise_prices(settings, synthetic=args.synthetic, persist=True)
    report = asyncio.run(
        generate_independent(
            settings=settings,
            prices=prices,
            provider_factory=_factory(settings, mock=args.mock),
            store=open_store(settings),
        )
    )
    print(report.plan.describe(), file=out)
    print(
        f"generated {report.generated}, skipped {report.skipped}, failed {report.failures}",
        file=out,
    )
    for model, failures in report.failures_by_model:
        print(f"  {model}: {failures} failed", file=out)
    return _generation_exit(report.generated, report.failures)


def _generation_exit(generated: int, failures: int) -> int:
    """A sweep in which nothing succeeded is a failure, not a result.

    Individual failures are data -- the rate per model is published. A run where
    *every* generation failed is an outage, and exiting 0 on it lets a wrapper
    script move on to the debate arms over a control arm that is flat everywhere.
    """
    if generated > 0 and failures == generated:
        _LOG.error("every generation failed; treating the run as an outage")
        return EXIT_FAILURE
    return EXIT_OK


def do_debate(args: argparse.Namespace, out: TextIO) -> int:
    """Run the three debate arms over the contested points."""
    settings = settings_from(args)
    pin_device(settings.cuda_visible_devices)
    prices = load_or_synthesise_prices(settings, synthetic=args.synthetic, persist=True)
    store = open_store(settings)
    decisions = stored_decisions(store)
    if decisions.empty:
        raise ValueError("no decisions are stored; run `generate` before `debate`")
    _refuse_a_half_swept_control(settings, prices, store)

    contested = select_contested(decisions, settings=settings)
    print(f"{len(contested)} contested point(s) to debate", file=out)
    chosen = getattr(args, "arms", None)
    arms = (
        TREATMENT_ARMS
        if not chosen
        else tuple(arm for arm in TREATMENT_ARMS if str(arm) in set(chosen))
    )
    report = asyncio.run(
        run_debate_arms(
            settings=settings,
            prices=prices,
            decisions=decisions,
            contested=contested,
            provider_factory=_factory(settings, mock=args.mock),
            store=store,
            arms=arms,
        )
    )
    print(
        f"{report.offered_points - report.dropped_points} point(s) answered by every "
        f"arm; {report.dropped_points} withheld from all three for want of a placebo "
        f"donor at least {settings.placebo_min_gap_sessions} session(s) back, holding "
        f"every seat, for each of {settings.max_debate_rounds} round(s)",
        file=out,
    )
    print(
        f"conversations {report.conversations}: held {report.held}, "
        f"skipped {report.skipped}, abandoned {report.abandoned}",
        file=out,
    )
    print(f"generated {report.generated} rows, {report.failures} failed", file=out)
    return _generation_exit(report.generated, report.failures)


def _refuse_a_half_swept_control(
    settings: Settings, prices: pd.DataFrame, store: DecisionStore
) -> None:
    """Refuse to pick a contested set from a control arm that has not finished.

    :func:`council.planning.plan_experiment` already discards a contested set
    measured over an unfinished control -- "a measurement of the half rather than a
    sample" -- and falls back to the assumed share. This command measured the same
    set from the same store with no such check and then committed the GPU to it.

    Dispersion is computed over whichever agents exist, and the independent sweep
    runs model, then persona, then ticker, so an interruption leaves a slice rather
    than a sample: :func:`~council.evaluation.dispersion.is_contested` rests on a
    directional split, and a slice missing the reversion personas collapses it. The
    debate arm would then be run over a non-randomly selected subset of the
    calendar -- and at the extreme the command prints "0 contested point(s) to
    debate", generates nothing and exits 0, which is what a finished run looks like.
    """
    run_plan = GenerationRunner(settings=settings, prices=prices, store=store).plan()
    if run_plan.remaining > 0:
        raise ValueError(
            f"the control arm is unfinished: {run_plan.remaining} of {run_plan.total} "
            "inference(s) remain. A contested set measured over a half-swept control "
            "measures the half, and the sweep runs model then persona then ticker, so "
            "what is missing is a slice rather than a sample. Finish `generate` first"
        )


def do_evaluate(args: argparse.Namespace, out: TextIO) -> int:
    """Score stored decisions, write the results object, print the summary."""
    settings = settings_from(args)
    prices = load_or_synthesise_prices(settings, synthetic=args.synthetic, persist=True)
    results = evaluate_experiment(
        settings=settings,
        prices=prices,
        decisions=stored_decisions(open_store(settings)),
        rule_name=args.rule,
        window_count=args.windows,
    )
    target = args.out or settings.data_dir / RESULTS_FILENAME
    write_results(results_as_json(results), target)
    print(render_results(results), file=out)
    print(f"\nwritten to {target}", file=out)
    return EXIT_OK


def do_probe(args: argparse.Namespace, out: TextIO) -> int:
    """Put the capitulation corpus to one model and score what contradicting it did."""
    settings = settings_from(args)
    pin_device(settings.cuda_visible_devices)
    run = asyncio.run(
        probe_model(
            settings=settings,
            provider_factory=_factory(settings, mock=args.mock),
            model=args.model,
            target=args.out,
        )
    )
    print(render_probe(run.report, model=run.model), file=out)
    print(f"\nwritten to {run.archive}", file=out)
    return _generation_exit(len(run.turns), run.failures)


def do_dryrun(args: argparse.Namespace, out: TextIO) -> int:
    """The whole pipeline, on synthetic prices and the mock provider.

    All four stages in the order a developer runs them, ``plan`` included. It costs
    nothing to issue and it is the stage most likely to break unnoticed -- nothing
    downstream reads its output -- so leaving it out would mean the one command
    whose job is to rehearse the pipeline left a quarter of it unexercised.

    The synthesised prices are written out first. Nothing else persists them, so
    this -- the only documented no-GPU path from nothing to a full set of results
    -- otherwise finishes with ``prices.parquet`` absent, and the dashboard
    reports that no run exists over a directory holding a complete one.
    Persisting them also makes the dry run replayable from disk rather than only
    from the seed.
    """
    settings = dryrun_settings(args)
    written = write_prices(
        load_or_synthesise_prices(settings, synthetic=True), settings.prices_path
    )
    print(f"synthetic prices written to {written}", file=out)
    # `--seconds-per-inference` belongs to the plan subparser alone, so the dry run
    # supplies the default the same way argparse would have.
    plan_args = argparse.Namespace(
        **{
            **vars(args),
            "synthetic": True,
            "mock": True,
            "seconds_per_inference": SECONDS_PER_INFERENCE,
        }
    )
    for step in (do_plan, do_generate, do_debate):
        code = step(_with(plan_args, settings), out)
        if code != EXIT_OK:
            return code
    return do_evaluate(_with(plan_args, settings), out)


def _with(args: argparse.Namespace, settings: Settings) -> argparse.Namespace:
    """Freeze the dry run's resolved settings onto the namespace the steps read.

    The steps rebuild settings from the namespace, so handing them the dry run's
    short calendar and private directory has to happen through the same arguments
    a user would have typed -- otherwise the dry run and a real run would take
    different paths through :func:`settings_from`, and the dry run would stop
    exercising the thing it is meant to rehearse.
    """
    return argparse.Namespace(
        **{
            **vars(args),
            "tickers": list(settings.tickers),
            "models": list(settings.agent_models),
            "start": settings.start,
            "end": settings.end,
            "data_dir": settings.data_dir,
            # Carried like the rest: the steps rebuild settings from this
            # namespace, so a value resolved here and not passed on is a value
            # the later stages never see. The dry run's reduced gap was exactly
            # that, and every placebo conversation in it was abandoned.
            "placebo_min_gap": settings.placebo_min_gap_sessions,
        }
    )


def write_results(payload: dict[str, Any], target: Path) -> None:
    """Write the published artefact as JSON every reader can parse.

    Raises:
        ValueError: if a non-finite float reached here.
            :func:`~council.report.results_as_json` renders those as their names,
            so one arriving as a float is a defect in that function rather than a
            file to write: ``allow_nan`` would otherwise emit an ``Infinity`` token
            no strict parser accepts, and the artefact would be readable only by
            the language that wrote it.
    """
    # Sorted keys and an explicit newline, so two runs of one configuration produce
    # byte-identical artefacts on any platform -- matching the completions archive.
    # Rendered before the file is opened, so a refusal leaves no half-written one.
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text + "\n")


# -- argument parsing -------------------------------------------------------------


def _common() -> argparse.ArgumentParser:
    """The overrides every subcommand accepts, in one place."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--tickers", nargs="+", metavar="SYMBOL", help="override the universe")
    parent.add_argument("--models", nargs="+", metavar="TAG", help="override the base models")
    parent.add_argument("--start", type=date.fromisoformat, metavar="YYYY-MM-DD")
    parent.add_argument("--end", type=date.fromisoformat, metavar="YYYY-MM-DD")
    parent.add_argument(
        "--device",
        metavar="INDEX",
        help="CUDA_VISIBLE_DEVICES for this process; an already-running Ollama "
        "daemon keeps the devices it was started with",
    )
    parent.add_argument("--data-dir", type=Path, metavar="PATH")
    parent.add_argument(
        "--placebo-min-gap",
        type=int,
        metavar="SESSIONS",
        help="how far back a placebo donor must come from. The default is a full "
        "lookback window, so the donor's data cannot overlap the day being "
        "decided; lowering it weakens the control",
    )
    parent.add_argument(
        "--synthetic",
        action="store_true",
        help="use generated prices instead of the configured parquet",
    )
    # Constrained rather than free text: logging.basicConfig runs before the try
    # block in main, so a typo here would raise a traceback instead of the one-line
    # message that block exists to produce -- and a typo is the likeliest failure a
    # user causes on this flag. `type` uppercases first, so `--log-level debug`
    # still works.
    parent.add_argument(
        "--log-level", default="INFO", type=str.upper, choices=LOG_LEVELS, metavar="LEVEL"
    )
    return parent


def _generating() -> argparse.ArgumentParser:
    """The extra flag only the two commands that call a model accept.

    Separate from :func:`_common` so that ``plan`` and ``evaluate`` do not offer a
    backend switch they never read -- an accepted flag that does nothing is a flag
    somebody will pass and believe.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--mock", action="store_true", help="answer with the mock provider; no daemon, no GPU"
    )
    return parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="council", description="A controlled study of persuasion between language models."
    )
    common = _common()
    generating = _generating()
    subcommands = parser.add_subparsers(dest="command", required=True)

    plan = subcommands.add_parser("plan", parents=[common], help="cost a configuration")
    plan.add_argument("--seconds-per-inference", type=float, default=SECONDS_PER_INFERENCE)
    plan.set_defaults(handler=do_plan)

    subcommands.add_parser(
        "generate", parents=[common, generating], help="run the independent arm"
    ).set_defaults(handler=do_generate)

    debate = subcommands.add_parser(
        "debate",
        parents=[common, generating],
        help="run the debate arms over contested points",
    )
    debate.add_argument(
        "--arms",
        metavar="ARM",
        nargs="+",
        choices=[str(arm) for arm in TREATMENT_ARMS],
        help=(
            "which treatment arms to run; defaults to all of them. The point set "
            "is unaffected -- servability is a property of the whole design, so a "
            "subset run answers the same points the full roster would"
        ),
    )
    debate.set_defaults(handler=do_debate)

    probe = subcommands.add_parser(
        "probe",
        parents=[common, generating],
        help="put the capitulation corpus to one model",
    )
    # One model per run: the report's headline is a property of a model, and the
    # sweep's --models list would average two of them into a number about neither.
    probe.add_argument(
        "--model", metavar="TAG", help="which model to probe; defaults to the first configured"
    )
    probe.add_argument("--out", type=Path, metavar="PATH", help="where the trials archive goes")
    probe.set_defaults(handler=do_probe)

    for name, handler, description in (
        ("evaluate", do_evaluate, "score stored decisions"),
        ("dryrun", do_dryrun, "the whole pipeline on the mock provider"),
    ):
        command = subcommands.add_parser(name, parents=[common], help=description)
        command.add_argument("--out", type=Path, metavar="PATH", help="where results.json goes")
        command.add_argument("--rule", default=PRIMARY_RULE, choices=sorted(RULES))
        command.add_argument("--windows", type=window_count, default=DEFAULT_WINDOW_COUNT)
        command.set_defaults(handler=handler)
    return parser


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    """Parse, dispatch, and turn an expected failure into a non-zero exit code.

    Only the failures this project knows how to explain are caught. A defect in the
    code should still print a traceback -- which also exits non-zero, so the
    guarantee that a failure never exits 0 does not depend on this list being
    complete.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    stream = sys.stdout if out is None else out
    try:
        exit_code: int = args.handler(args, stream)
    except (FileNotFoundError, ValueError, ProviderError) as failure:
        _LOG.error("%s: %s", type(failure).__name__, failure)
        return EXIT_FAILURE
    return exit_code
