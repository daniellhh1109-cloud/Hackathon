"""Export one month of label-free multimodal predictions; no automatic training."""
import argparse
import json
from pathlib import Path
import torch
import yaml
from src.inference.predict_multimodal import predict_month


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/multimodal_data.yaml')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--year',type=int,required=True)
    p.add_argument('--month',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--batch-size',type=int,default=128)
    args=p.parse_args();raw=yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(2)
    report=predict_month(raw['store_dir'],raw['cache_dir'],args.checkpoint,year=args.year,month=args.month,
                         output_dir=args.output_dir,batch_size=args.batch_size)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
