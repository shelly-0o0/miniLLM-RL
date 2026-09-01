# 来源、版本与代码归属

本文档回答三个问题：工程以哪个版本为基线、每个算法依据什么一手资料实现、哪些代码是 MiniMind 上游已有而不是本次原创。

## 1. 工程基线

- 唯一工程基线：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)，Apache-2.0 License。
- 本地开始开发时的提交：`393e387e9ad99f0f04c296e4c5e7353f4444629f`（2026-08-06）。
- 2026-08-29 复核的官方最新提交：`d65ef2c00ebc6082f9df11541e1b191655eddb00`。两个提交之间只有 README 变化，相关模型与训练代码一致。
- 上游已有的模型、数据集和训练器没有被宣称为本次原创。本次在官方代码之上补充统一策略目标、DAPO/GSPO、动态采样、RLVR verifier、实验日志、评测、消融和测试。

## 2. 架构与监督训练的一手来源

| 能力 | 一手资料 | 本项目中的归属 |
|---|---|---|
| Decoder-only Transformer | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | MiniMind 上游实现 |
| RMSNorm | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | MiniMind 上游实现 |
| RoPE | [RoFormer](https://arxiv.org/abs/2104.09864) | MiniMind 上游实现 |
| YaRN | [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071) | MiniMind 上游实现 |
| GQA | [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) | MiniMind 上游实现；本次增加 KV Cache 理论/实测脚本 |
| Flash Attention | [FlashAttention](https://arxiv.org/abs/2205.14135)、[PyTorch scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | 上游通过 PyTorch SDPA 进入可用后端；是否实际选中 Flash 内核由硬件、dtype、shape 和 PyTorch 后端决定 |
| SwiGLU | [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | MiniMind 上游实现 |
| 稀疏 MoE | [Switch Transformers](https://arxiv.org/abs/2101.03961) | MiniMind 上游 Top-1 路由与负载均衡辅助损失 |
| LoRA | [LoRA](https://arxiv.org/abs/2106.09685) | MiniMind 上游训练链路 |
| 知识蒸馏 | [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | MiniMind 上游 logits 蒸馏链路；本次补充 FP32 log-target KL、CE-only 同预算对照、独立 holdout 评测和不完整累积窗口修正 |
| AMP | [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) | MiniMind 上游实现 |
| DDP | [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) | MiniMind 上游实现；本次新增训练器沿用其初始化方式 |

注意：PyTorch SDPA 是统一接口，不等于在任意设备上都调用 Flash Attention。简历中应写“接入 SDPA/Flash Attention 路径并在目标 GPU 上验证后端与峰值显存”，不能仅凭源码调用就写“实现 FlashAttention-2”。

## 3. 策略优化算法来源与映射

| 算法/机制 | 一手论文或官方实现 | 本次实现位置 | 实现边界 |
|---|---|---|---|
| GRPO | [DeepSeekMath](https://arxiv.org/abs/2402.03300) | `trainer/policy_optimization.py`、`trainer/train_grpo.py`、`trainer/train_agent.py` | 保留 MiniMind 的组相对优势；增加固定旧策略、多轮 policy epoch 和可审计指标 |
| CISPO | [MiniMax-M1](https://arxiv.org/abs/2506.13585)、[MiniMax-M1 官方仓库](https://github.com/MiniMax-AI/MiniMax-M1) | `trainer/policy_optimization.py` | 按论文实现 token-level 全局均值、`1+ε_high^IS` 上界、stop-gradient 重要性权重；修正 MiniMind 上游把 CLI 值直接当绝对上界的语义，没有把论文中的系统级结果冒充本项目结果 |
| DAPO | [DAPO 论文](https://arxiv.org/abs/2503.14476)、[作者官方 DAPO 仓库](https://github.com/BytedTsinghua-SIA/DAPO) | `trainer/policy_optimization.py`、`trainer/train_agent.py` | 按论文式 (10)–(13) 实现 Clip-Higher、二值动态采样、token-level loss、soft overlong punishment；未照搬 verl 源码 |
| GSPO | [GSPO 论文](https://arxiv.org/abs/2507.18071)、[Alibaba ROLL 的 GSPO 配置示例](https://github.com/alibaba/ROLL/blob/main/examples/docs_examples/example_gspo.yaml) | `trainer/policy_optimization.py` | 按论文式 (5)–(7) 实现长度归一化序列重要性比率及序列裁剪；ROLL 仅用于交叉核对 `importance_sampling: seq` 配置语义；当前 advantage 是轨迹级，因此不是 GSPO-token 变体 |
| DPO（独立偏好分支） | [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | MiniMind 上游 `trainer/train_dpo.py` + 本次 `prepare_dpo_data.py` / `eval_dpo_holdout.py` | 目标函数为上游已有；本次补充 prompt-grouped holdout、同 token 物化评测、最终 remainder 更新顺序、原子保存和目标单测 |

DAPO 官方 recipe 的历史复现环境曾固定到 verl 提交 `4f80e465c2ec79ab9c3c30ec74b9745de61d0490`。本项目只参考论文定义和公开 recipe 的配置语义，在 MiniMind 张量形状与 rollout 结构中独立实现，没有复制第三方函数或源码片段。

## 4. Agentic RL / RLVR 来源与映射

| 内容 | 一手资料 | 本次实现位置 |
|---|---|---|
| 多轮工具调用强化学习范式 | [ReTool](https://arxiv.org/abs/2504.11536) | `trainer/train_agent.py` 的多轮 assistant → tool → observation rollout |
| 可扩展多工具环境与 rollout 设计参考 | [VerlTool 论文](https://arxiv.org/abs/2509.01055)、[VerlTool 官方仓库](https://github.com/TIGER-AI-Lab/verl-tool) | 用作设计参照；本项目仍使用 MiniMind 的确定性模拟工具，不引入其框架代码 |
| 结构化可验证奖励 | MiniMind 上游 Agent 数据和工具执行器 + 本次独立设计 | `calculate_rewards(..., return_details=True)`：分别记录格式、工具名、参数、执行、答案和二值任务成功 |
| 固定集评测与逐轨迹审计 | 本次独立实现 | `scripts/eval_agent_rlvr.py` |

当前工具环境是可复现的 calculator、单位换算、天气、时间、汇率和翻译 mock，不是联网搜索、容器代码执行或生产级浏览器沙箱。它足以验证策略目标、格式约束和多轮 credit assignment，但不能据此声称已解决真实开放世界 Agent 安全问题。

## 5. 本次新增代码的原创性声明

以下文件是在 MiniMind 接口上为本项目独立编写的教学复现，没有逐行复制论文作者或 verl/VerlTool 的实现：

- `trainer/policy_optimization.py`
- `trainer/experiment_logging.py`
- `scripts/eval_agent_rlvr.py`
- `scripts/split_agent_dataset.py`
- `scripts/summarize_eval_results.py`
- `scripts/summarize_rl_metrics.py`
- `scripts/run_rlvr_ablation.sh`
- `scripts/run_rlvr_eval.sh`
- `scripts/benchmark_architecture.py`
- `scripts/verify_flash_sdpa.py`
- `scripts/verify_model_flash_sdpa.py`
- `scripts/audit_moe_routing.py`
- `scripts/audit_sft_mask.py`
- `scripts/audit_lora_split.py`
- `scripts/audit_tool_rlvr_dataset.py`
- `scripts/audit_multiturn_serialization.py`
- `scripts/prepare_tool_rlvr_data.py`
- `scripts/prepare_tool_rlvr_challenge_data.py`
- `scripts/prepare_agent_sft_data.py`
- `scripts/run_agent_sft.sh`
- `scripts/prepare_kd_data.py`
- `scripts/run_student_pretrain.sh`
- `scripts/run_student_sft.sh`
- `scripts/run_kd_ablation.sh`
- `scripts/run_kd_pipeline_after_pretrain.sh`
- `scripts/eval_distillation.py`
- `scripts/eval_lora_holdout.py`
- `scripts/prepare_dpo_data.py`
- `scripts/eval_dpo_holdout.py`
- `scripts/collect_experiment_evidence.py`
- `scripts/run_dpo_after_readiness.sh`
- `scripts/run_rlvr_pilot_after_dpo.sh`
- `scripts/run_architecture_verification_after_pilot.sh`
- `scripts/run_rlvr_formal_after_architecture.sh`
- `scripts/run_finalize_after_formal.sh`
- `tests/test_distillation.py`
- `tests/test_dpo_objective.py`
- `tests/test_kv_cache_consistency.py`
- `tests/test_architecture_features.py`
- `tests/test_agent_sft_data.py`
- `tests/test_policy_optimization.py`
- `tests/test_agent_rewards.py`
- `learning_notebooks/08_Agentic_RLVR从零到多轮工具训练.ipynb`
- `learning_notebooks/09_GRPO_CISPO_DAPO_GSPO从公式到代码.ipynb`

`trainer/train_grpo.py` 与 `trainer/train_agent.py` 是在 MiniMind 原文件上的扩展。论文提供数学目标，官方 recipe 提供参数语义，最终代码根据 MiniMind 的 batch、mask、DDP 和 rollout 约定重新实现。

## 6. 数据、部署和工具来源

- 数据下载入口与数据集 ID以 [MiniMind 官方 README](https://github.com/jingyaogong/minimind) 为准。
- Hugging Face 下载命令以 [huggingface_hub CLI 文档](https://huggingface.co/docs/huggingface_hub/en/package_reference/cli) 为准。
- SGLang rollout 服务命令以 [SGLang 官方文档](https://docs.sglang.ai/) 为准。
- 第三方数据的许可证、用途限制、隐私与污染情况必须在正式训练前单独审查；仓库能下载不等于可以用于商业训练。
- `agent_rl_tool_verified_*` 不是外部数据集：它由 `prepare_tool_rlvr_data.py` 针对 MiniMind 现有确定性 mock 工具独立生成，并在 manifest 中记录类别、train/eval 问题哈希零交叉和 SHA-256。原因是上游 `agent_rl.jsonl` 的后半部分为无工具、无 GT 的普通对话，不能冒充严格 Tool-Use RLVR 数据。
- `agent_rl_tool_challenge_*` 也不是外部数据：基础 Tool 数据在 Agent SFT 后实测饱和，因而由 `prepare_tool_rlvr_challenge_data.py` 独立生成六类未进入冷启动标签的组合任务。它只复用同一个确定性工具环境，oracle 字段仅用于全量重放审计，不会由 RL loader 提供给策略。
- 计算器执行器由本项目从 Python `eval` 改为 AST 白名单解释器；这是安全加固代码，不源自论文或第三方仓库。它仍不是 OS 级生产沙箱。

## 7. 可引用与不可引用的结果

- 可以引用：自己运行产生的配置、日志、checkpoint、逐样本轨迹和三随机种子聚合表。
- 可以作为背景引用：论文作者报告的数字，但必须明确说“论文报告”，不能写成“本项目提升”。
- 不可引用：上游 README 截图、单次 cherry-pick 指标、随机小模型 smoke test 或未保存配置的手工观察。
- RTX 4090 上已产生的真实结果、未完成项和能力边界见 `05_ACTUAL_EXPERIMENT_RESULTS.md`；没有产物的队列任务不会被写成已完成实验。
