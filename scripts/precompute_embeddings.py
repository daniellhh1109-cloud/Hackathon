"""Audit, encode and resume the local filing embedding cache."""
import argparse
import json
from pathlib import Path
import sys

from filelock import Timeout
import yaml
from src.data.precompute_text_embeddings import read_config,run_pipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/text_embeddings.yaml')
    parser.add_argument('--max-documents',type=int,help='Bound newly attempted unique texts; reports remain explicitly partial')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--allow-download',action='store_true',help='Download public model files only; filings stay local')
    parser.add_argument('--show-monthly',action='store_true',help='Include by_filing_month in terminal JSON; the saved report always includes it')
    parser.add_argument('--debug',action='store_true',help='Preserve tracebacks for operational failures')
    args=parser.parse_args()
    if args.max_documents is not None and args.max_documents<1:parser.error('--max-documents must be positive')
    try:
        raw,config=read_config(args.config)
        report=run_pipeline(raw['source_path'],raw['cache_dir'],config,model_cache_dir=raw['model_cache_dir'],
            max_documents=args.max_documents,audit_only=args.audit_only,retry_failed=args.retry_failed,allow_download=args.allow_download)
    except (OSError, ValueError, RuntimeError, Timeout, yaml.YAMLError) as exc:
        if args.debug:
            raise
        parser.exit(1, f'Encoding failed ({type(exc).__name__}): {exc}\nUse --debug for the full traceback.\n')
    # The pipeline contract is a dictionary, never an exception object or a partial fallback.
    if not isinstance(report, dict):
        parser.exit(1, f'Pipeline contract error: expected a report dictionary, got {type(report).__name__}.\n')
    display = report if args.show_monthly else {k:v for k,v in report.items() if k!='by_filing_month'}
    print(json.dumps(display,indent=2,allow_nan=False))
    report_path=(Path(raw['cache_dir'])/'coverage_report.json').resolve()
    print(f'Full report (including monthly statistics): {report_path}', file=sys.stderr)


if __name__=='__main__':main()
