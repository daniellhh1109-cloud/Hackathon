# Guide architecture and implementation status

The project now follows Section 12 of the technical guide in directory structure. **This is scaffolding, not implementation of ModernTCN/MiniLM/Agent.**

## Clean start
Old research implementation and its tests were removed at the user's request. MAIN.py is a nonfunctional entry point that exits explicitly. No legacy research results establish correctness of this scaffold.

## Planned modules
The docstring in each new `src/` module describes its responsibility. No placeholder returns fake predictions or weights. `tests/test_alignment.py`, `test_no_leakage.py`, `test_constraints.py`, `test_model_shapes.py` are marked planned and contain no test claims.

`configs/base.yaml` preserves guide defaults, with a project-relative factor-list path. It is NOT yet consumed by MAIN.py. Dependencies for PyTorch, MiniLM and CVXPY will be selected when those modules are implemented.

## Module map
- `src/data/`: quant preprocessing, filing cache, sample windows, annual splits.
- `src/models/`: ModernTCN, filing/time attention, gated fusion.
- `src/training/`: annual fitting, validation, prediction metrics.
- `src/inference/`: monthly predictions.
- `src/portfolio/`: optimizer, risk checks, holdings-first backtest.
- `src/agent/`: tools, prompts, bounded controller.
- `src/utils/`: common I/O and calendar helpers.
- `data/` and `outputs/`: local-only content, versioned empty-directory placeholders.

## Non-negotiable interfaces
Target month is feature month plus exactly one calendar month. Never shift ret_exc_lead1m again. Use the official 147-factor whitelist. Quant windows contain 12 continuous calendar months of one security; text covers six eligible historical months with masks. Lock holdings before reading their realized returns. Candidate count is not nonzero holdings count. All-empty text requires explicit handling.

## Team ownership
A: integration and rules. B: data audit/schema. C: timing and leakage tests. D: baseline review, then numerical model. E: portfolio checks/backtest. Coordinate text and Agent ownership before Weeks 3-4.

## Next steps
First complete Week 1 checks; then implement and test modules incrementally. Implement the root entry point only after the modules have validated behavior.
