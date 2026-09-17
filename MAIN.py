"""Reproducible stage entrypoint; training/inference keep their existing CLIs."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import subprocess
import sys

STAGES = {
    'train-quant': 'scripts.run',
    'train-multimodal': 'scripts.run_multimodal',
    'predict-quant': 'scripts.predict',
    'predict-multimodal': 'scripts.predict_multimodal',
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    project_stages = {'setup','doctor','audit','prepare','embed','train','predict','backtest','report','all'}
    if argv and argv[0] == 'demo':
        from scripts.week4_demo import main as demo_main
        return demo_main(argv[1:])
    if argv and argv[0] in project_stages:
        from scripts.project import main as project_main
        return project_main(argv)
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='Portable commands: setup, doctor, audit, prepare, embed, train, predict, backtest, report, all. Pass --help after a stage to see its existing options. Portfolio smoke uses synthetic fixtures only.')
    parser.add_argument('stage', choices=[*STAGES, 'portfolio'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    remaining = args.arguments
    if args.stage == 'portfolio':
        from src.agent.pipeline import main as portfolio_main
        return portfolio_main(remaining)
    command = [sys.executable, '-m', STAGES[args.stage], *remaining]
    return subprocess.run(command, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())

