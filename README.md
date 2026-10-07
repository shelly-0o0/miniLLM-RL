# miniLLM-RL

一个面向 GSM8K 数学 Agentic RL 后训练的可复现实验项目。

本项目以 MiniMind-64M 作为 Stage 1 基座，使用统一的 GSM8K 数据、calculator 环境和数学 verifier，研究 PPO、GRPO、CISPO、DAPO、GSPO 的 Agentic RL 后训练；Stage 2 将同一套多轮工具环境迁移到 Qwen3-4B-Base。完成的 Track 2 主线以互斥 A/B 数据比较 Agent-SFT、GRPO 和 Additional-SFT；Track 1 的 SFT → GRPO 已完成，Pure GRPO 曾在 42,904 条 cold-start rollout 后中断，现已从持久 checkpoint 恢复并补齐 Base/SFT-only/Pure/SFT→GRPO 四臂评测。项目重点是验证训练闭环、奖励信号、冷启动作用和跨规模迁移，不以 GSM8K SOTA 为目标。

## 项目阶段

```text
Stage 1: MiniMind-64M → Agent SFT → PPO/GRPO/CISPO/DAPO/GSPO
Stage 2 Track 1: Qwen3-4B-Base → Base / Pure GRPO / SFT only / SFT → GRPO
Stage 2 Track 2: Agent-SFT(A) → GRPO(A) / GRPO(B) / Additional-SFT(B)
```

同一阶段的比较使用相同 GSM8K manifest、prompt、calculator、answer parser、rollout 参数和固定评测集。Stage 2 的 Pure GRPO 从 Base 的新建零初始化 LoRA 开始，SFT → GRPO 从 SFT adapter 开始，以隔离 warm start 的作用。

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
5. [Stage 1 GSM8K 最终报告](docs/STAGE1_GSM8K_FINAL_REPORT.md)
6. [Stage 2 完整搭建与算法记录](docs/STAGE2_QWEN3_4B_BUILD_LOG.md)
7. [Stage 2 Track 2 最终报告](docs/STAGE2_TRACK2_FINAL_REPORT.md)
8. [项目总报告：算法、过程、结果与结论](docs/PROJECT_SUMMARY_REPORT.md)
9. [Stage 2 Track 1（Stack 1）实验设计报告](docs/STAGE2_TRACK1_EXPERIMENT_REPORT.md)
10. [Stage 2 Track 1 Pure GRPO 中断与恢复记录](docs/STAGE2_TRACK1_TERMINATION_REPORT.md)
11. [可发布实验结果索引](results/README.md)

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

准备 GSM8K 固定切分并生成可审计 manifest：

```bash
python scripts/download/download_gsm8k.py
python scripts/prepare/prepare_gsm8k.py
python scripts/prepare/prepare_gsm8k_agent_data.py
python scripts/evaluate/evaluate_gsm8k.py \
  --references data/processed/gsm8k/validation.jsonl \
  --predictions path/to/predictions.jsonl
```

预测文件每行至少包含 `id` 和 `prediction`。训练、验证和官方测试文件会进行题目级泄漏检查；manifest 记录每个 split 的行数与 SHA-256。

## Stage 2：Qwen3-4B Agentic GRPO

Stage 2 的额外依赖、无权重 readiness 审计和 smoke test：

```bash
python -m pip install -r requirements-stage2.txt
python scripts/audit_qwen_stage2.py

bash scripts/run_qwen_stage2.sh smoke_sft
bash scripts/run_qwen_stage2.sh smoke_pure_grpo
bash scripts/run_qwen_stage2.sh smoke_sft_grpo
```

smoke 全部通过后再运行正式矩阵：

```bash
bash scripts/run_qwen_stage2.sh sft
bash scripts/run_qwen_stage2.sh pilot_pure_grpo
bash scripts/run_qwen_stage2.sh pilot_sft_grpo
bash scripts/run_qwen_stage2.sh pure_grpo
bash scripts/run_qwen_stage2.sh sft_grpo
bash scripts/run_qwen_stage2.sh eval_validation
bash scripts/run_qwen_stage2.sh eval_test
```

训练入口保留真实的 `calculate_math` 调用、工具执行结果回填、第二轮生成和严格 RLVR verifier，不是只约束答案文本格式的单轮 GRPO。完整原理、显存设计、日志字段、恢复规则和执行顺序见 [Stage 2 完整记录](docs/STAGE2_QWEN3_4B_BUILD_LOG.md)。

Track 1 的 SFT→GRPO 已完成并在冻结 test shard 上达到 67.475% strict accuracy。Pure GRPO 曾在 5,363/6,726 组时中断，最后 200 组 strict/tool execution 均为 0；现已从第 5,350 组 checkpoint 恢复。Base 和 SFT-only 的缺失评测已并行启动，Pure 完成后将自动合并并审计完整四臂结果。研究问题、四臂对照、两阶段 SFT、共同 GRPO 协议、评测和审计门禁见 [Track 1 实验设计报告](docs/STAGE2_TRACK1_EXPERIMENT_REPORT.md)；中断快照和恢复边界见 [Pure GRPO 中断与恢复记录](docs/STAGE2_TRACK1_TERMINATION_REPORT.md)。

### Track 2：互斥 A/B 数据实验

Track 2 从 Agent-SFT(A) 的同一 adapter 分叉，比较 GRPO(A)、GRPO(B) 与 Additional-SFT(B)，用于区分已见题 RL、新题 RL 和继续监督学习。先运行：

```bash
bash scripts/run_qwen_track2.sh prepare
bash scripts/run_qwen_track2.sh audit
bash scripts/run_qwen_track2.sh smoke_sft_a
bash scripts/run_qwen_track2.sh probe_smoke
bash scripts/run_qwen_track2.sh smoke_additional_sft_b
bash scripts/run_qwen_track2.sh smoke_grpo_a
bash scripts/run_qwen_track2.sh smoke_grpo_b
```

数据边界、算法公式、完整执行顺序、监控和结论限制见 [Track 2 完整记录](docs/TRACK2_AB_AGENTIC_RL_BUILD_LOG.md)。

Track 2 已完成五臂 1,319 题官方测试并通过 fail-closed 审计。严格任务成功率为 Base 0%、Agent-SFT(A) 0.910%、GRPO(A) 2.578%、GRPO(B) 2.654%、Additional-SFT(B) 42.532%。完整配对统计与结论边界见 [Track 2 最终报告](docs/STAGE2_TRACK2_FINAL_REPORT.md)。

### SVAMP 跨数据集 warm-start GRPO

SVAMP 实验从 Track 2 的 `Additional-SFT(B)` adapter 出发，只使用 SVAMP 作者提供的交叉验证文件：fold 0–3 构成 816 题训练集，fold 4 构成 184 题冻结 holdout，并按 `group_nums` 检查变体家族无交叉。

```bash
bash scripts/run_svamp_warm_grpo.sh download
bash scripts/run_svamp_warm_grpo.sh prepare
bash scripts/run_svamp_warm_grpo.sh audit
bash scripts/run_svamp_warm_grpo.sh probe
bash scripts/run_svamp_warm_grpo.sh eval_zero_shot
bash scripts/run_svamp_warm_grpo.sh grpo
bash scripts/run_svamp_warm_grpo.sh eval_grpo
bash scripts/run_svamp_warm_grpo.sh merge_eval
bash scripts/run_svamp_warm_grpo.sh audit_results
```

该协议是从作者 CV folds 派生的项目内 train/holdout，不应写成 SVAMP 官方 train/test。源 revision、行数与 SHA-256 固定在 `dataset/manifests/svamp_agent.json`。

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
