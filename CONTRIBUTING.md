# How to challenge this study

This is a finished research artefact, not a library looking for features. The analysis is
frozen at the tagged release, and the most useful thing anyone can do with it is **try to
break it**.

Six rounds of adversarial audit already changed the headline finding twice, withdrew a claim
the data did not license, and falsified a mechanism sentence that the entire test suite had
endorsed. Every one of those is recorded in [`docs/CLAIMS.md`](docs/CLAIMS.md) rather than
quietly corrected. A seventh round from someone with no stake in the result is worth more
than any feature.

## What is most welcome

**A challenge to a numbered claim.** Every claim in [`docs/CLAIMS.md`](docs/CLAIMS.md) has an
identifier, and each is either backed by an artefact in this repository or explicitly marked
as not backed. If a claim overreaches its evidence, name the claim and say what the evidence
actually supports. Claims C8, C29 and C30 have all been rewritten this way already.

**A reproduction failure.** Every published figure is recomputed from the pinned artefacts by
`tests/test_docs_findings.py`, and the suite runs on CPU with no model server:

```bash
uv sync --frozen --extra dev --extra app
uv run pytest
```

Both extras, not just `dev`: the suite imports the dashboard layer, so `dev` alone fails at
collection rather than at a test.

If that suite is green on your machine and a number in the documents still disagrees with
what you compute from `docs/results/`, that is a defect worth an issue. Include what you
computed and the code you computed it with.

**A confound the design does not control.** This is how the two extension arms came to exist:
the placebo could not separate "reacting to contradiction" from "reacting to an argument that
cannot be reconciled" (D8), so an arm was built that could. If you can name a rival
explanation the six arms do not rule out, say what design would rule it out.

**A defect in the instrument.** D15 — the inference backend not enforcing the numeric schema
bounds the contradictor arm relied on — was invisible to every test because the mock provider
obeyed a constraint the real backend ignored. Defects of that shape are the ones this
register most wants and is least able to find on its own.

## What is out of scope

Re-running the experiment to get different numbers is not a correction: the backend drifts
between versions, which is registered as D16, and a rerun disagreeing with the published
series is expected rather than informative. Style, structure and naming changes to a frozen
artefact are also out of scope — the code is what produced the numbers, and it stays that way.

## If you do open a pull request

Everything must pass before it can land, and all three run without a GPU:

```bash
uv run pytest
uv run ruff check .
uv run mypy src tests
```

A change that moves a published number must also update the document that quotes it, and
add an entry to the defect register saying what moved and whether any conclusion changed.
That is the standard the existing entries are held to.
