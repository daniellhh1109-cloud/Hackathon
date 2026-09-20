# 按功能命名的代码结构

2026-09-20整理。文件名按功能区分，不以成员字母或开发周次区分。

- 数据准备与审计：`scripts/run_data.py`。
- 数值训练与比较：`src/training/quant_components.py`、`scripts/verify_quant_integration.py`、`scripts/evaluate_quant.py`。
- 线性基线：`scripts/run_ridge_baseline.py`、`scripts/run_official.py`。
- 组合与演示：`scripts/verify_portfolio.py`、`scripts/demo_pipeline.py`。
- 试跑/评价配置：`configs/ridge_smoke.json`、`configs/linear_smoke.json`、`configs/quant_evaluation.yaml`。
- 功能测试：`test_quant_training.py`、`test_quant_dataset.py`、`test_multimodal_training.py`、`test_agent_pipeline.py`、`test_linear_baseline.py`、`test_quant_evaluation.py`、`test_pipeline.py`。

重复的trainer/dataset/全年线性配置统一到 `trainer.yaml`、`datasets.yaml`、`official_full.json`。数值训练、数据准备、官方线性训练、预测检查和持仓检查各保留一个实现入口，删除纯转发副本。数据目录内的旧encoder副本移除；唯一实现为 `src/data/precompute_text_embeddings.py`。引用失效接口、仅针对早期固定样例的时间线验收脚本、没有测试内容的占位文件和过时的骨架交接说明已移除。

依赖文件保留四个不同用途：完整环境、已选择PyTorch后的公共环境、固定版本线性基线、仅文本编码。源代码、当前测试、规则、数据目录说明以及不同算法实现均保留。

历史运行目录不重命名。旧缓存/模型仍可通过显式配置引用；新运行使用功能化输出路径。模型结构与checkpoint格式不因文件整理而改变。
