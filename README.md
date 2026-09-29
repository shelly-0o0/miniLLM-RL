# miniLLM-RL

一个面向 GSM8K 数学 Agentic RL 后训练的可复现实验项目。

本项目以 MiniMind-64M 作为 Stage 1 基座，使用统一的 GSM8K 数据、calculator 环境和数学 verifier，研究 PPO、GRPO、CISPO、DAPO、GSPO 的 Agentic RL 后训练；Stage 2 计划将同一套数据和评测协议迁移到 Qwen3-4B，执行 LoRA SFT 与 RL。项目重点是验证训练闭环、奖励信号、算法稳定性和跨规模迁移，不以 GSM8K SOTA 为目标。

## 项目阶段

```text
Stage 1: MiniMind-64M → Agent SFT → PPO/GRPO/CISPO/DAPO/GSPO
Stage 2: Qwen3-4B   → LoRA Agent SFT → GRPO/PPO
```

所有 RL 方法从同一个 SFT checkpoint 分叉，使用相同 GSM8K manifest、prompt、calculator、answer parser 和固定评测集。

## 项目内容

- 从零开始的 Dense/MoE 轻量语言模型训练链路
- Pretrain、Assistant-only SFT、LoRA、知识蒸馏和 DPO
- 多轮 Tool-Use Agent 环境与 Agentic RLVR
- GRPO、CISPO、DAPO、GSPO 统一策略优化实现
- 训练集/评测集按题目分组隔离，降低数据泄漏风险
- 可验证奖励、Action Mask、轨迹序列化和行为策略概率审计
- 固定集、多随机种子评测和本地 JSONL 实验记录
- Flash SDPA、KV Cache、GQA、MoE 路由等架构验证

## 项目结构

```text
miniLLM-RL/
├── model/                     # 模型结构、LoRA、Tokenizer
├── trainer/                   # Pretrain/SFT/DPO/RL/蒸馏训练代码
├── scripts/                   # 数据准备、训练流水线、评测与审计脚本
├── tests/                     # 模型、算法、verifier 和数据测试
├── docs/foundation_model_interview/
│   ├── 00_END_TO_END_GUIDE.md
│   ├── 01_TRAINING_PIPELINE_REPORT.md
│   ├── 02_AGENTIC_RLVR_FROM_ZERO_TO_MASTERY.md
│   ├── 03_POLICY_OPTIMIZATION_FROM_ZERO_TO_MASTERY.md
│   ├── 04_EXPERIMENT_PROTOCOL.md
│   ├── 05_ACTUAL_EXPERIMENT_RESULTS.md
│   ├── 06_FINAL_PROJECT_REPORT.md
│   ├── 08_EXPERIMENTAL_NARRATIVE.md
│   └── 09_RL_OPTIMIZATION_FOLLOWUP.md
├── learning_notebooks/        # 可执行学习与实验 Notebook
├── patches/                   # 性能优化补丁
└── requirements.txt
```

## 新主线文档

1. [项目计划](docs/PROJECT_PLAN.md)
2. [数据集与 Benchmark 规范](docs/DATASET_AND_BENCHMARK.md)
3. [实验协议](docs/EXPERIMENT_PROTOCOL.md)
4. [工作记录](docs/WORKLOG.md)

旧的 MiniMind/RL 研究报告保留在 `docs/foundation_model_interview/`，作为历史实现和实验审计材料，不代表新的 GSM8K 主线已经完成。

## 历史文档阅读顺序

1. [复现包总入口](docs/foundation_model_interview/README.md)
2. [端到端执行指南](docs/foundation_model_interview/00_END_TO_END_GUIDE.md)
3. [训练链路报告](docs/foundation_model_interview/01_TRAINING_PIPELINE_REPORT.md)
4. [Agentic RLVR 从零到代码](docs/foundation_model_interview/02_AGENTIC_RLVR_FROM_ZERO_TO_MASTERY.md)
5. [GRPO/CISPO/DAPO/GSPO 从零到代码](docs/foundation_model_interview/03_POLICY_OPTIMIZATION_FROM_ZERO_TO_MASTERY.md)
6. [实验与数据真实性规范](docs/foundation_model_interview/04_EXPERIMENT_PROTOCOL.md)
7. [实际实验结果](docs/foundation_model_interview/05_ACTUAL_EXPERIMENT_RESULTS.md)
8. [最终项目报告](docs/foundation_model_interview/06_FINAL_PROJECT_REPORT.md)
9. [完整实验叙事](docs/foundation_model_interview/08_EXPERIMENTAL_NARRATIVE.md)
10. [RL 优化后续报告](docs/foundation_model_interview/09_RL_OPTIMIZATION_FOLLOWUP.md)

## 快速开始

```bash
git clone https://github.com/shelly-0o0/miniLLM-RL.git
cd miniLLM-RL

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch 需要根据本机 CUDA、操作系统和 GPU 环境单独安装。完整数据集和模型 checkpoint 不放入 Git 仓库，请先阅读 [数据说明](dataset/dataset.md) 和 [端到端执行指南](docs/foundation_model_interview/00_END_TO_END_GUIDE.md)。

运行测试：

```bash
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py'
```

## 我的主要工作

- `trainer/policy_optimization.py`：统一实现 GRPO、CISPO、DAPO 和 GSPO 的目标函数与诊断指标。
- `trainer/train_agent.py`、`trainer/rollout_engine.py`：多轮工具调用轨迹、可验证奖励和 Agentic RLVR 训练闭环。
- `scripts/`：数据划分、冷启动数据构造、固定集评测、架构核验和实验结果汇总。
- `tests/`：覆盖 KV Cache、MoE、DPO、蒸馏、策略目标、奖励函数和数据处理的回归测试。
- `docs/foundation_model_interview/`：记录实现依据、实验协议、真实结果、失败实验和结论边界。

项目报告中保留了真实实验结果和负结果；所有数字应以对应报告和实验产物为准，不将上游 MiniMind 的结果冒充为本项目结果。

## 上游项目与许可证

本项目基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 开展，MiniMind 是本项目的模型与训练基线。上游代码许可证见 [LICENSE](LICENSE)，本项目新增内容与来源映射见 [SOURCES.md](docs/foundation_model_interview/SOURCES.md)。

欢迎通过 Issue 讨论问题，也欢迎针对代码、实验复现和文档提出建议。
