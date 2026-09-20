"""Compatibility imports; the complete orchestration now lives in MAIN.py."""
from MAIN import (save_yaml, setup, doctor, audit, prepare, embed, train, predict,
                  backtest, run_pipeline, project_main as main)

if __name__ == '__main__':
    raise SystemExit(main())
