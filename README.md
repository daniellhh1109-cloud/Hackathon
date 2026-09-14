# McGill-FIAM 2026 Hackathon

五人团队按技术指南第 12 节建立的干净开发骨架。旧研究代码已按用户要求移除。**当前未实现预测模型、Agent、优化器或回测；没有可报告的模型测试结果。**

## 目录

```text
configs/base.yaml          指南默认参数
src/data/                  数值预处理、文本向量、样本与年度划分
src/models/                ModernTCN、文本 attention、多模态融合
src/training/              trainer、预测指标
src/inference/             每月预测
src/portfolio/             优化器、风险检查、回测
src/agent/                 工具、提示词、调用循环
src/utils/                 I/O、日期辅助
tests/                     四类待编写测试
MAIN.py                    总入口（未实现，默认退出并说明）
data/raw/                  原始比赛文件（仅本地）
data/processed/            预处理数据（仅本地）
data/embeddings/           文本向量（仅本地）
outputs/                   checkpoints、predictions、holdings、returns、reports
```

## 开始开发

```bash
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
python MAIN.py --help
```

当前 `python MAIN.py` 会明确返回“尚未实现”，不会生成假结果。测试文件目前是职责占位，`pytest` 尚无测试可收集，不等于验收通过。`configs/base.yaml` 是后续实现的配置规格，当前入口尚未加载或执行它。

## 开发顺序

1. B 审计数据；A 确认官方规则与团队默认值。
2. C 实现目标月份对齐、年度划分、防泄漏反例测试。
3. D 完成数值预处理与基线，然后逐步实现 ModernTCN。
4. 实现冻结 MiniLM、月度/六个月 attention 和门控融合。
5. E 实现优化器与约束；随后接入 Agent 和先锁持仓后评价的回测。
6. 每一阶段通过实际测试后才更新完成状态。

## 数据与协作

- [本地数据](data/README.md)
- [架构与接口](docs/architecture.md)
- [协作流程](CONTRIBUTING.md)
- [第一周分工](docs/week1/integration.md)
- [规则记录](competition_rules.md)：需继续核对原始官方材料。
- [提交清单](submission_checklist.md)

仅同步代码、配置和团队文档。原始数据、密钥、PDF/CV、embedding、模型文件、研究输出和旧提交草稿均不上传。本地 `.venv` 如已有安装属于个人环境，不代表项目已锁定依赖。
