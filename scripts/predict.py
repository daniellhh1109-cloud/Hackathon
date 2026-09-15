"""Export one month or a completed checkpoint's full OOS year without reading labels."""
import argparse
from pathlib import Path
import torch
import yaml
from src.inference.predict_month import MonthlyPredictor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/inference.yaml')
    parser.add_argument('--month',help='Optional YYYY-MM; omitted exports the annual test interval')
    parser.add_argument('--output-dir',help='Override with a NEW output directory')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    if type(config['cpu_threads']) is not int or config['cpu_threads']<1:
        raise ValueError('cpu_threads must be positive')
    torch.set_num_threads(config['cpu_threads'])
    data=yaml.safe_load(Path(config['dataset_config']).read_text())
    predictor=MonthlyPredictor(data['store_dir'],config['checkpoint'],year=config['year'],batch_size=config['batch_size'])
    print(predictor.export(args.output_dir or config['output_dir'],month=args.month))


if __name__=='__main__': main()
