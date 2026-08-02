# AI Agent Instructions for `gaia-symbolic-regression`

## Purpose
This repository explores symbolic regression on Gaia DR3 data to recover a stellar flux relation comparable to Stefan–Boltzmann and inverse-square law theory.

## Key files
- `README.md` — project overview and objectives
- `requirements.txt` — install dependencies
- `src/model.py` — primary Python model code
- `notebooks/` — analysis and feature-engineering notebooks
- `data/` — source data files; do not create or commit large raw data files unless the user explicitly requests it

## Recommended workflow for Git-related requests
- Base all new work on the `main` branch.
- Use descriptive feature branches such as `feature/<topic>`, `bugfix/<topic>`, or `docs/<topic>`.
- Keep changes small and focused.
- Do not modify notebooks or add large files without explicit user approval.
- Before proposing code changes, verify whether the user asked for a code fix, feature, or only a git/branching recommendation.

## Development guidance
- Install dependencies with:
  - `python -m pip install -r requirements.txt`
- Validate Python files with compilation or linting before suggesting changes.
- There is no dedicated test suite or CI configuration present in the repository.
- When code changes are requested, prefer modifying `src/` and keep notebook edits minimal unless explicitly asked.

## Notes for AI agents
- Link to existing documentation instead of duplicating it: use `README.md` for project goals and objectives.
- If the user asks for git workflow help, provide branch naming, commit scope, and verification steps rather than making arbitrary repository changes.
- Avoid introducing new project conventions unless the user requests them.
