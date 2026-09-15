"""Audit, encode and resume the local filing embedding cache."""
import argparse
import json
from src.data.precompute_text_embeddings import read_config,run_pipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/text_embeddings.yaml')
    parser.add_argument('--max-documents',type=int,help='Bound newly attempted unique texts; reports remain explicitly partial')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--allow-download',action='store_true',help='Download public model files only; filings stay local')
    args=parser.parse_args()
    if args.max_documents is not None and args.max_documents<1:parser.error('--max-documents must be positive')
    raw,config=read_config(args.config)
    report=run_pipeline(raw['source_path'],raw['cache_dir'],config,model_cache_dir=raw['model_cache_dir'],
        max_documents=args.max_documents,audit_only=args.audit_only,retry_failed=args.retry_failed,allow_download=args.allow_download)
    print(json.dumps({k:v for k,v in report.items() if k!='by_filing_month'},indent=2))

if __name__=='__main__':main()
