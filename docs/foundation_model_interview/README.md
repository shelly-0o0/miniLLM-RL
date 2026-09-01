# MiniMind 大模型基座岗位复现包

这套材料以官方 MiniMind 仓库为唯一工程基线，目标是把“看过代码”升级为“能解释、能运行、能量化、能接受项目拷问”。

## 建议阅读顺序

1. [端到端执行指南](./00_END_TO_END_GUIDE.md)：服务器、环境、数据、Pretrain、SFT、LoRA、蒸馏、Agent/RLVR、四算法训练、评测与故障处理。
2. [训练链路面试报告](./01_TRAINING_PIPELINE_REPORT.md)：Pretrain、Assistant-only SFT、LoRA、蒸馏和训练稳定性。
3. [Agentic RL / RLVR 从零到可运行代码](./02_AGENTIC_RLVR_FROM_ZERO_TO_MASTERY.md)：先建立轨迹、工具环境、action mask 和可信 verifier。
4. [GRPO、CISPO、DAPO、GSPO 从零到代码精通](./03_POLICY_OPTIMIZATION_FROM_ZERO_TO_MASTERY.md)：再使用同一环境比较策略目标。
5. [实验与数据真实性规范](./04_EXPERIMENT_PROTOCOL.md)：如何产生可以写进简历的真实量化结果。
6. [本机真实实验结果](./05_ACTUAL_EXPERIMENT_RESULTS.md)：4090 实验的配置、数字、失败边界与可用的面试口径。
7. [最终项目总报告](./06_FINAL_PROJECT_REPORT.md)：把实际工作流、实现、结果、复盘和面试表述串成一条主线。
8. [简历对齐技术报告](./07_RESUME_ALIGNED_TECHNICAL_REPORT.md)：按四条简历要求逐项讲解核心代码、数学原理、DPO 离线偏好分支、四种在线策略算法、真实结果、负结果、结论边界和高频项目拷问。
9. [实验叙事报告](./08_EXPERIMENTAL_NARRATIVE.md)：按照真实发生顺序串联实验目的、执行过程、问题定位、修复、新模型和最终结果。
10. [RL 不增长诊断与后续优化](./09_RL_OPTIMIZATION_FOLLOWUP.md)：定位 checkpoint 精度、行为策略概率和稀疏奖励问题，记录 Tool 冷启动、严格 DAPO 与数学后续优化的真实结果。
11. [来源与代码归属](./SOURCES.md)：论文、官方代码和本次新增代码的逐项映射。

这不是两个互相独立的 RL 项目。正确依赖关系是：

```text
通用 SFT
  → 训练集 oracle 轨迹的 Agent SFT 冷启动
  → Agent 环境与多轮 trajectory
  → 可验证 reward 与 action mask
  → GRPO/CISPO/DAPO/GSPO 策略更新
  → 固定集评测与三 seed 统计
```

配套代码实验为 [Agent/RLVR Notebook](../../learning_notebooks/08_Agentic_RLVR从零到多轮工具训练.ipynb)、[四算法 Notebook](../../learning_notebooks/09_GRPO_CISPO_DAPO_GSPO从公式到代码.ipynb) 与 [简历项目完整工作报告 Notebook](../../learning_notebooks/10_简历项目完整工作报告.ipynb)，用于观察真实函数、tensor、梯度、verifier 输出和最终机器证据。

## 先说结论：官方已有与本次新增

| 需求 | 官方 MiniMind 已有 | 本次补充 | 面试时应如何表述 |
|---|---|---|---|
| Decoder-only 与现代模块 | RMSNorm、RoPE/YaRN、GQA、KV Cache、SwiGLU、PyTorch SDPA/Flash Attention 路径、Top-1 稀疏 MoE | 架构量化脚本 | “阅读、验证并做性能量化”，不要说这些上游模块都是自己原创 |
| Pretrain/SFT/LoRA/KD | 四条训练脚本、Assistant-only mask、AMP、梯度累积、DDP、断点续训 | 完整操作顺序、审计项与面试报告 | 可以说“复现并排查训练链路”，具体指出自己验证过的实现细节 |
| GRPO/CISPO | 已有，但原实现只有两种 loss 且指标较少 | 统一 loss、严格 CISPO token-mean、DAPO、GSPO、旧策略多轮更新、正 KL、clip/ratio 指标 | 这是本次最主要的算法开发工作 |
| DAPO | 无 | 非对称 Clip、二值动态采样缓冲、token-level loss、软超长惩罚 | 必须同时打开相应参数才可称为完整 DAPO |
| GSPO | 无 | 长度归一化序列重要性比率、序列裁剪、序列级优化 | 当前实现是标准 GSPO；同轨迹 token advantage 时可扩展 GSPO-token |
| Agentic RL/RLVR | 已有多轮 rollout、模拟工具与 GRPO/CISPO | 结构化 verifier、任务成功/格式/参数/执行/最终答案分离指标、DAPO/GSPO 接入、固定集评测 | 强调这是小规模可控环境，不冒充生产级浏览器/代码沙箱 |
| 真实量化 | 上游 README 有上游曲线 | JSONL 日志、固定集评测、三 seed 聚合、架构 benchmark | 只把自己机器生成的 CSV 数字写成个人结果 |

## 本次新增或实质修改的文件

- `trainer/policy_optimization.py`：四种策略目标与 DAPO 工具函数。
- `trainer/experiment_logging.py`：本地 JSONL 指标。
- `trainer/train_grpo.py`：接入 DAPO/GSPO、多轮 policy update、统一指标和逐批策略同步。
- `trainer/train_agent.py`：接入四算法、动态采样、软超长惩罚和可验证奖励。
- `scripts/eval_agent_rlvr.py`：固定测试集的逐轨迹证据与汇总结果。
- `scripts/split_agent_dataset.py`：按问题哈希持久化无精确题目泄漏的训练/评测划分。
- `scripts/prepare_tool_rlvr_data.py`：构建与本地工具环境一致、训练/评测问题隔离的可执行 Tool-Use RLVR 数据。
- `scripts/prepare_agent_sft_data.py` 与 `run_agent_sft.sh`：用训练集 oracle 轨迹做 Agent 协议冷启动，不使用评测标签。
- `scripts/summarize_eval_results.py`：先合并解码 seed，再跨训练 seed 聚合固定集结果。
- `scripts/summarize_rl_metrics.py`：跨随机种子聚合。
- `scripts/run_rlvr_ablation.sh`：四算法 × 三随机种子实验矩阵。
- `scripts/run_math_dapo_optimization.sh` 与 `run_math_optimization_eval.sh`：FP32 数学 DAPO 后续优化和固定 128 条 holdout × 三解码种子评测。
- `scripts/run_tool_corrected_ablation.sh` 与 `run_tool_corrected_ablation_eval.sh`：从同一 Tool 课程 SFT 补跑修复后的 GRPO/CISPO/GSPO，并与 DAPO 做统一固定集对照。
- `scripts/benchmark_architecture.py`：KV Cache、GQA 显存公式、解码吞吐和峰值显存。
- `scripts/audit_sft_mask.py`：逐样本核验 Assistant-only mask 并保存可读监督片段。
- `tests/`：算法与 verifier 单元测试。
- `docs/foundation_model_interview/02_AGENTIC_RLVR_FROM_ZERO_TO_MASTERY.md`：问题驱动的 Agent/RLVR 入门到进阶教程。
- `docs/foundation_model_interview/03_POLICY_OPTIMIZATION_FROM_ZERO_TO_MASTERY.md`：四算法公式、tensor、代码和实验诊断教程。
- `learning_notebooks/08_*`、`09_*`：直接导入真实仓库函数的可执行实验课。

## 真实性边界

本项目已在单张 RTX 4090 上完成 Dense Pretrain、通用 SFT、LoRA、KD、DPO、Agent SFT、部分 MoE 验证，以及数学/组合 Tool-Use 的四算法 Formal。完整初始数字见 [本机真实实验结果](./05_ACTUAL_EXPERIMENT_RESULTS.md)：LoRA 在独立医疗 holdout 上取得正结果，KD 与 DPO 是如实保留的负结果，最初两套 RLVR 在 `10 updates / 1e-7` 小预算下都未改变固定集指标。后续定位并修复 checkpoint 精度、行为 log-prob 对齐和 Tool 冷启动问题；96 条独立 Tool challenge holdout、三个解码种子下，严格成功率由 4.86% 提至课程 SFT 的 35.76%。同一课程初始化与 30-update 目标下，GRPO/CISPO/DAPO/GSPO 分别为 35.76%/35.07%/36.11%/35.76%；主要收益来自无 holdout 泄漏的课程 SFT，DAPO 独立增量只有 +0.35pp 且 rollout 成本约 9.33 倍，详见 [后续优化报告](./09_RL_OPTIMIZATION_FOLLOWUP.md)。MoE 只完成 10.52% 单 epoch 的短程训练，因此只支撑路由、参数和系统代价结论，不支撑质量优于 Dense。
