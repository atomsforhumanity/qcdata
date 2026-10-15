# AGENTS.md

Instructions for AI coding agents (and a quick reference for people) working in `qcdata`. CONTRIBUTING.md explains the same workflow in more detail.

## This package

`qcdata` defines the shared vocabulary of the suite: pydantic models for structures, program inputs (`ProgramInput`, `ProgramSpec`, `FileInput`), results (`SinglePointData`, `OptimizationData`, ...), and the `CalcType` enum.

- Changes here ripple through every other package. Add fields and models deliberately, keep them backward compatible where possible, and call out breaking changes in CHANGELOG.md.
- Results are in atomic units (Bohr, Hartree) unless a field name says otherwise; document the unit of every new numeric field.
- New calculation types need a `CalcType` value, a result model, and an entry in the `get_data_type` mapping.

## The qc* suite

Dependencies point one way: qcref -> qcdata -> {qcinf, qccodec} -> qccompute. Never import a package that sits later in that order, and keep changes that span packages in separate pull requests, released in that order.

## Commands

```sh
uv sync --all-extras --all-groups     # install everything
uv run pre-commit install             # once per clone: run the checks on every commit
uv run pytest                         # tests
uv run ruff check --fix . && uv run ruff format .
uv run mypy
uv run pre-commit run --all-files     # exactly what CI checks
```

Use `uv add` / `uv remove` for dependencies; never edit `uv.lock` by hand.

## Code conventions

- Use type annotations for all function and method parameters (except `self` and `cls`) and return values, including `-> None`. mypy enforces this for library code.
- Annotate class and instance attributes, including dataclass and pydantic fields. Annotate local variables only when inference is insufficient.
- Prefer precise types; avoid `Any` unless genuinely necessary.
- Write Google-style docstrings for all public modules, classes, functions, and methods (ruff enforces this). Start with a one-line summary ending in a period; add Args, Returns, Yields, and Raises sections where applicable; describe meaning and units without repeating types already in annotations.
- Use `X | None`, not `Optional[X]`. Target Python 3.10.
- Keep changes focused. Do not reformat or refactor unrelated code.

## Before you finish

- `uv run pre-commit run --all-files` and `uv run pytest` pass.
- New behavior has tests; bug fixes have a test that failed before the fix.
- User-visible changes are listed under `## [unreleased]` in CHANGELOG.md.
- `pyproject.toml` has no `[tool.uv.sources]` path or git entries for qc* packages (CI rejects them).
- Do not commit, push, tag, or release unless asked to.
