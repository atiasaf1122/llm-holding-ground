"""What the peer is allowed to say, and which item's argument it says it with.

Two properties carry the whole design and neither is visible in a finished record:
the peer contradicts whatever the model said, and the placebo differs from the real
challenge in which question the argument is about and in nothing else. Both are
pure functions of the corpus, so both are asserted against values here rather than
against a run.
"""

from __future__ import annotations

import pytest

from council.probe.challenge import (
    Condition,
    build_challenge,
    contradiction,
    select_placebo_donor,
)
from council.probe.items import Verdict, load_items
from helpers_probe import CAPITAL, CORPUS, ELEMENT, PLANET, SEED

# -- who the peer argues against ----------------------------------------------


def test_the_contradiction_always_asserts_the_option_the_model_did_not_choose() -> None:
    assert contradiction(CAPITAL, Verdict.CORRECT) == (
        CAPITAL.distractor,
        CAPITAL.distractor_argument,
    )
    assert contradiction(CAPITAL, Verdict.OTHER) == (CAPITAL.answer, CAPITAL.answer_argument)


# -- the placebo --------------------------------------------------------------


def test_the_placebo_makes_the_same_claim_and_only_changes_the_argument() -> None:
    real = build_challenge(item=CAPITAL, verdict=Verdict.CORRECT, condition=Condition.CHALLENGE)
    sham = build_challenge(
        item=CAPITAL, verdict=Verdict.CORRECT, condition=Condition.PLACEBO, donor=PLANET
    )

    assert sham.claim == real.claim
    assert sham.label == real.label
    assert sham.argument != real.argument


def test_the_placebo_argument_comes_from_another_item_in_the_corpus() -> None:
    donor = select_placebo_donor(item=CAPITAL, donors=CORPUS, seed=SEED)
    sham = build_challenge(
        item=CAPITAL, verdict=Verdict.CORRECT, condition=Condition.PLACEBO, donor=donor
    )

    assert sham.argument in {ELEMENT.distractor_argument, PLANET.distractor_argument}


def test_the_placebo_borrows_the_side_the_real_condition_would_have_argued() -> None:
    # Borrowing distractor_argument regardless moved relevance and truth-flavour
    # together: a model that opened wrong met a sound case in the real condition and
    # a deliberately fallacious one in the control, so the placebo rate fell for a
    # reason that is not irrelevance and the difference read as persuasion.
    held = build_challenge(
        item=CAPITAL, verdict=Verdict.CORRECT, condition=Condition.PLACEBO, donor=PLANET
    )
    missed = build_challenge(
        item=CAPITAL, verdict=Verdict.DISTRACTOR, condition=Condition.PLACEBO, donor=PLANET
    )

    assert held.argument == PLANET.distractor_argument
    assert missed.argument == PLANET.answer_argument


def test_a_donor_is_never_the_item_being_probed() -> None:
    for item in CORPUS:
        assert select_placebo_donor(item=item, donors=CORPUS, seed=SEED) is not item


def test_reordering_the_pool_cannot_change_the_donor() -> None:
    first = select_placebo_donor(item=CAPITAL, donors=CORPUS, seed=SEED)
    second = select_placebo_donor(item=CAPITAL, donors=CORPUS[::-1], seed=SEED)

    assert first is second


def test_removing_an_item_nobody_drew_cannot_change_a_single_draw() -> None:
    # The guarantee the docstring makes: the draw depends on the seed and the two
    # identifiers and on nothing else. An index taken modulo the number of
    # candidates re-draws almost every donor whenever the pool changes size, so a
    # rerun over a subset quietly rewrites trials that were already recorded -- and
    # the gpu-marked smoke test runs exactly such a subset.
    corpus = load_items()
    drawn = {
        item.identifier: select_placebo_donor(item=item, donors=corpus, seed=SEED)
        for item in corpus
    }

    for removed in corpus:
        pool = tuple(item for item in corpus if item is not removed)
        for item in pool:
            if drawn[item.identifier] is removed:
                continue
            assert (
                select_placebo_donor(item=item, donors=pool, seed=SEED)
                is drawn[item.identifier]
            ), f"{item.identifier} moved when {removed.identifier} left"


def test_a_subset_that_still_holds_the_donors_re_draws_none_of_them() -> None:
    corpus = load_items()
    probed = corpus[:5]
    donors = {select_placebo_donor(item=item, donors=corpus, seed=SEED) for item in probed}
    subset = tuple(sorted({*probed, *donors}, key=lambda entry: entry.identifier))

    for item in probed:
        assert select_placebo_donor(item=item, donors=subset, seed=SEED) is select_placebo_donor(
            item=item, donors=corpus, seed=SEED
        ), item.identifier


def test_a_different_seed_can_draw_a_different_donor() -> None:
    drawn = {
        select_placebo_donor(item=CAPITAL, donors=CORPUS, seed=seed).identifier
        for seed in range(40)
    }

    assert drawn == {ELEMENT.identifier, PLANET.identifier}


def test_a_placebo_with_nobody_else_to_borrow_from_is_refused() -> None:
    # Falling back to the item's own argument would make the placebo a second copy
    # of the real condition wearing the control's label.
    with pytest.raises(ValueError, match="no placebo donor"):
        select_placebo_donor(item=CAPITAL, donors=(CAPITAL,), seed=SEED)


def test_a_placebo_challenge_without_a_donor_is_refused_rather_than_falling_back() -> None:
    with pytest.raises(ValueError, match="no placebo donor"):
        build_challenge(item=CAPITAL, verdict=Verdict.CORRECT, condition=Condition.PLACEBO)
