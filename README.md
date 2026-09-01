# miniLLM-RL

一个面向大模型基座训练与 Agentic RLVR 的可复现实验项目。

本项目以 [MiniMind](https://github.com/jingyaogong/minimind) 作为轻量级语言模型基线，重点展示我在训练链路增强、策略优化、工具调用、可验证奖励、固定集评测和实验复盘方面的工程实现与实验结果。这里的重点不是重新介绍上游项目，而是呈现一套可以阅读、运行、审计和量化的个人项目成果。

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

## 推荐阅读顺序

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
