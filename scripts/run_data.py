"""Prepare full-universe quant store and audit annual loaders/real timelines."""
import argparse
from pathlib import Path
import yaml
from src.data.splits import validate_quant_window

from src.data.quant_dataset import prepare_store, QuantDataset, make_loader, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/member_b_week2.yaml')
    parser.add_argument('--reuse', action='store_true', help='Use existing immutable prepared store')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if not args.reuse:
        prepare_store(config['raw_path'], config['factor_path'], config['store_dir'])
    report_dir = Path(config['report_dir'])
    report_dir.mkdir(parents=True, exist_ok=True)
    audits = {}
    for partition in ['train', 'validation', 'test']:
        dataset = QuantDataset(config['store_dir'], year=config['year'], partition=partition)
        if not len(dataset):
            raise ValueError(f'empty {partition} dataset')
        batch = next(iter(make_loader(dataset, shuffle=partition == 'train')))
        audits[partition] = {**dataset.audit, 'batch_quant_shape': list(batch['quant'].shape),
                             'batch_has_target': 'target' in batch}
        if partition == 'test':
            for permno in [12490, 14593, 10107]:
                matches = dataset.samples.index[(dataset.samples.permno == permno) &
                                                 (dataset.samples.target_month == f"{config['year']}-01")]
                if len(matches):
                    timeline = dataset.timeline(int(matches[0]))
                    records = timeline.to_dict('records')
                    for row in records:
                        row['target_month'] += '-01'  # calendar guards use full dates; A uses YYYY-MM
                    validate_quant_window(records, permno, f"{config['year']}-01-01")
                    timeline.to_csv(report_dir / f'timeline_{permno}.csv', index=False)
    write_json(report_dir / 'loader_audit.json', audits)
    print(audits)


if __name__ == '__main__':
    main()
