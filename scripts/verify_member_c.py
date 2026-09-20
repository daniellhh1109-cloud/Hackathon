"""Run from repository root: python -m scripts.verify_member_c --audit-dir PATH."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from src.data.splits import (annual_split, validate_quant_window, sample_label,
                            select_filings, validate_feature_columns)
from src.utils.dates import filing_available_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('docs/week1/member_c_timeline.md'))
    args = parser.parse_args()
    def read(name):
        with (args.audit_dir/name).open() as f:
            return list(csv.DictReader(f))
    quant = read('ibm_2021_numeric_timeline.csv')
    filings = read('ibm_2021_filing_timeline.csv')
    factors = read('verified_factors.csv')
    names = [r['variable'] for r in factors]
    validate_feature_columns(names, names)
    target, permno = '2021-01-01', '12490'
    rows = validate_quant_window(quant, permno, target)
    allowed = select_filings(filings, permno, target)
    ids = {r['document_id'] for r in allowed}
    # Cross-check issuer/share-class with audited numeric rows in the filing month.
    by_month = {r['eom'][:7]: r for r in rows}
    for f in allowed:
        q = by_month[f['filing_date'][:7]]
        if (f['gvkey'], f['iid']) != (q['gvkey'], q['iid']):
            raise ValueError('GVKEY/IID mismatch')
    assert len(rows) == 12 and len(allowed) == 5
    assert sample_label(quant, permno, target) == '-0.053827'
    report = ['# 成员 C：IBM 真实时间线验收', '',
        '目标：2021-01；PERMNO：12490；GVKEY：006066；IID：01。', '',
        '来源是成员 B 对本地原始 Parquet 的审计导出，脚本不改写原始数据。', '',
        '截止时间为 2021-01-01 00:00 UTC（不含）。公告可得时间采用供应商 filing_date + 3 个日历日；这是团队保守假设，不是 SEC 接受时间认证。', '',
        '## 数值窗口', '', '| 特征月末 | 源观测日 | 该行标签对应月份 | ret_exc_lead1m |', '|---|---|---|---|']
    report += [f"| {r['eom']} | {r['date']} | {r['target_month']} | {r['ret_exc_lead1m']} |" for r in rows]
    report += ['', '**目标标签直接取 2020-12 行的 −0.053827（−5.3827% 下一月超额收益），不再 shift；该数值仅用于事后训练/评价，不能作为预测输入。**', '',
        '## 公告窗口与排除证据', '', '| document_id | 供应商日期 | 假设可得日期 UTC | 2021-01 可用 |', '|---|---|---|---|']
    report += [f"| {r['document_id']} | {r['filing_date']} | {filing_available_at(r['filing_date']).date()} | {'是' if r['document_id'] in ids else '否'} |" for r in sorted(filings, key=lambda r:r['filing_date'])]
    report += ['', '六个月指 2020-07 至 2020-12 的发布日期窗口，并非必须每月都有公告。8 条候选中 5 条合格；12 月 31 日公告延迟至 1 月 3 日，两条 1 月公告也被排除。文档 ID 中的日期不能替代 filing_date。合格公告的 GVKEY/IID 已与同月数值记录交叉核对。', '',
        '## 年度划分（全部为目标收益月，包含端点）', '', '| 年份 | 训练 | 验证 | 测试 |', '|---|---|---|---|']
    for year in range(2021, 2027):
        s = annual_split(year)
        report.append(f'| {year} | {s.train_start:%Y-%m}–{s.train_end:%Y-%m} | {s.validation_start:%Y-%m}–{s.validation_end:%Y-%m} | {s.test_start:%Y-%m}–{s.test_end:%Y-%m} |')
    report += ['', '共 68 个样本外月份。12 月历史要求会减少早期实际可训练样本，但不会改变官方划分边界，也不能跳过缺失的样本外月份。', '', '## 输入 SHA-256', '']
    for name in ['ibm_2021_numeric_timeline.csv', 'ibm_2021_filing_timeline.csv', 'verified_factors.csv']:
        report.append(f"- `{name}`：`{hashlib.sha256((args.audit_dir/name).read_bytes()).hexdigest()}`")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(quant_months=len(rows), eligible_filings=len(allowed), rejected_filings=len(filings)-len(allowed), factors=len(names), label=sample_label(quant, permno, target), report=str(args.output)), ensure_ascii=False))

if __name__ == '__main__':
    main()
