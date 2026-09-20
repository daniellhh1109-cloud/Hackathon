# McGill-FIAM 2026 Hackathon

按功能组织的数值/多模态训练、月度推理、组合优化和回测项目。统一入口是 `MAIN.py`，源码在 `src/`，命令在 `scripts/`，配置在 `configs/`，测试在 `tests/`。

## 安装与检查

先按 PyTorch 官网安装适合本机的 CPU/CUDA wheel，再安装公共依赖：

```bash
python -m pip install -r requirements-portable.txt
python MAIN.py doctor
python -m pytest -q
```

如果不需要单独选择 PyTorch 安装渠道，可使用 `requirements.txt`，它包含公共依赖和 PyTorch。`requirements-baseline.txt` 保留历史线性基线的固定版本；`requirements-text-embeddings.txt` 用于仅运行文本编码的环境。

## 训练与推理

两个组委会 Parquet 放在 `data/raw/`；147因子名单在 `configs/factor_char_list.csv`。

```bash
python MAIN.py setup --device cpu --data-dir data/raw
python MAIN.py audit
python MAIN.py prepare
python MAIN.py train --kind quant --year 2021
python MAIN.py predict --kind quant --year 2021
python MAIN.py embed --allow-download
python MAIN.py train --kind multimodal --year 2021
python MAIN.py predict --kind multimodal --year 2021
```

GPU 在 setup 时选择 `--device cuda`。多模态训练要求完整文本缓存。依次完成2022–2026年度模型；已有实验需在配置中显式指定原缓存和 checkpoint 路径，历史输出不自动迁移。

## 按功能使用工具

|任务|入口/配置|
|---|---|
|数值缓存和数据加载审计|`python -m scripts.run_data --config configs/datasets.yaml`|
|独立Ridge试跑|`python -m scripts.run_ridge_baseline --config configs/ridge_smoke.json`|
|官方线性模型全年实验|`python -m scripts.run_official --config configs/official_full.json`|
|官方线性模型小规模验收|`configs/linear_smoke.json`|
|数值模型/基线比较|`python -m scripts.evaluate_quant --help`、`configs/quant_evaluation.yaml`|
|数值模型整合验收|`python -m scripts.verify_quant_integration --help`|
|组合手算验收|`python -m scripts.verify_portfolio`|
|保存预测的独立校验|`python -m scripts.check_predictions --help`|
|持仓约束检查|`python check_constraints.py HOLDINGS.csv --require-full-period`|
|68个月合成整合演示|`python MAIN.py demo --output-dir outputs/demo_new`|

持仓CSV的 `WEIGHT` 为NAV百分数（1代表1%），内部优化权重为小数（0.01代表1%）。真实回测遇到持仓收益缺失会停止。合成演示和单轮训练不代表正式完整策略成绩；全量文本缓存、正式全期回测、Deck与CV仍需完成。

[架构与接口](docs/architecture.md) · [目录清理说明](docs/code_layout.md) · [官方规则](competition_rules.md) · [提交清单](submission_checklist.md)
