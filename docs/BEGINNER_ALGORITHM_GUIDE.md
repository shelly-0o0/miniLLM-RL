# Mini-RL 初学者算法与代码导读

这份文档假设你已经知道 Transformer、token、prompt、SFT 和“强化学习通过奖励更新模型”的基本概念，但还不熟悉完整训练工程。

## 1. 这个项目究竟在研究什么

项目不是要从零训练一个能达到业界 SOTA 的数学大模型，而是要回答一个更适合教学和实验的问题：

> 同一份数学数据、同一个工具环境、同一个初始模型和同一个评测标准下，PPO、GRPO、CISPO、DAPO、GSPO 这些强化学习方法有什么差异？它们能否先在 64M 小模型上跑通，再迁移到 Qwen3-4B？

主线分两阶段：

```text
Stage 1：MiniMind-64M
Base → Agent SFT → PPO / GRPO / CISPO / DAPO / GSPO → 固定集评测

Stage 2：Qwen3-4B
Base → LoRA Agent SFT → GRPO / PPO（可选 DAPO）→ 同协议评测
```

当前状态：Stage 1 的数据、SFT、rollout、奖励、统一策略目标、评测和大部分测试已经存在，并产生过本地 SFT/GRPO pilot 产物；Stage 2 仍是计划，没有真正的 Qwen3 adapter 和训练入口。

## 2. 先理解一条数据如何流过系统

一条 GSM8K 题目大致经历以下流程：

```text
原始 question + answer
  ↓ 数据规范化与固定切分
训练集 / 验证集 / 测试集
  ↓ Agent 模板化
system + user(question) + assistant(tool call / reasoning / answer)
  ↓ Agent SFT
学会输出工具调用格式和最终答案格式
  ↓ RL rollout
同一道题采样多个不同回答
  ↓ calculator + verifier
得到 0/1 奖励及格式、工具、答案等诊断指标
  ↓ PPO 或 GRPO-family loss
提高成功轨迹概率，降低失败轨迹概率
  ↓ 固定 holdout 评测
比较 Base、SFT 和各 RL checkpoint
```

这里有三个必须分开的概念：

1. **SFT 学格式和基本行为**：如果模型连工具 JSON 或最终答案都不会输出，稀疏的 0/1 RL 奖励几乎无法学习。
2. **RL 做相对优化**：从同一个 SFT checkpoint 分叉，避免前一个算法改变后一个算法的起点。
3. **评测不参与训练**：validation 用于选择配置，test 只用于最终报告。

## 3. 数据层：如何保证实验可信

### 3.1 原始数据与固定切分

`scripts/download/download_gsm8k.py` 下载官方 GSM8K。`dataset/gsm8k.py` 是数据规则核心：

- 把不同来源字段统一成项目格式；
- 用固定 seed 划分 train/validation；
- 保留官方 test；
- 对标准化题目计算 hash，检查各 split 没有重复题；
- 给输出文件计算 SHA-256，生成 manifest。

`scripts/prepare/prepare_gsm8k.py` 是命令行入口，调用上述函数生成 `data/processed/gsm8k/*.jsonl` 和 `dataset/manifests/gsm8k.json`。

设计原因：如果每次运行都随机切分，算法 A 和算法 B 看到的数据可能不同；如果同一道题同时进入训练和测试，准确率就不可信。

### 3.2 Agent 数据

`scripts/prepare/prepare_gsm8k_agent_data.py` 将普通数学问答改造成 Agent 轨迹。它把题目转换成：

```text
用户题目 → assistant 决定是否调用 calculator → tool 返回结果 → assistant 给最终答案
```

训练数据又分为：

- `train_sft.jsonl`：有标准行为示例，用于冷启动 SFT；
- `train_rl.jsonl`：只给任务和可验证答案，让策略自己 rollout；
- `validation_rl.jsonl`、`test_rl.jsonl`：固定评测。

`dataset/lm_dataset.py` 把 JSONL 转换成 token、label 和 mask。对 Agent SFT，最重要的是 **assistant-only mask**：用户问题和工具 observation 是上下文，不应让模型学习去预测它们；loss 只落在 assistant 输出 token 上。

`scripts/audit_sft_mask.py` 和 `tests/test_agent_sft_data.py` 用来证明 mask 真的符合这个约定。

### 3.3 数学环境与 verifier

`trainer/math_env.py` 同时被训练和评测调用：

- 安全解析算术表达式；
- 用 AST 白名单执行 calculator，不直接使用危险的 Python `eval`；
- 从 `#### 42`、`answer: 42`、`\boxed{42}` 等文本提取答案；
- 将整数、小数、分数归一化后比较。

这是 RLVR 的关键。RLVR 是 Reinforcement Learning with Verifiable Rewards：不需要主观 reward model，而是用确定性规则判断答案是否正确。训练和评测必须复用同一套 parser/verifier，否则会出现“训练认为正确、评测认为错误”的漂移。

`scripts/evaluate/evaluate_gsm8k.py` 是离线预测文件评测入口；`tests/test_gsm8k_pipeline.py` 覆盖切分、泄漏、calculator 和答案比较。

## 4. 模型层：MiniMind 在做什么

`model/model_minimind.py` 是 MiniMind 主模型，来自上游项目。可以把它理解成缩小版的现代 Decoder-only LLM：

- token embedding 把 token ID 变成向量；
- 多层 causal self-attention 只看当前位置之前的 token；
- RoPE 提供位置信息；
- GQA 减少 KV cache；
- RMSNorm 稳定激活；
- SwiGLU/MoE 提供非线性变换；
- LM head 预测下一个 token；
- generation 循环不断采样下一个 token。

`model/model_lora.py` 是上游自定义 LoRA。LoRA 冻结原始大矩阵，只训练两个低秩小矩阵。它减少可训练参数，但不必然大幅降低所有激活显存。这个实现只面向 MiniMind，不等于 Hugging Face PEFT，也不能直接用于 Qwen3。

`trainer/trainer_utils.py` 是所有训练器的底座：

- 初始化 DDP 和当前 GPU；
- 固定随机种子；
- 创建模型与 tokenizer；
- 计算学习率；
- 保存模型权重和完整 resume checkpoint；
- 处理 rank 0 日志与原子写入。

DDP 的原理是每张卡复制完整模型、处理不同数据，再同步梯度。因此它提高吞吐量，但不会让单卡放不下的模型自动分片。

## 5. 监督阶段：为什么 RL 前必须 SFT

### 5.1 Pretrain

`trainer/train_pretrain.py` 做 next-token prediction：给定前面的 token，预测下一个 token。它建立基本语言能力，但不知道 Agent 格式或 GSM8K 工具协议。

`scripts/run_student_pretrain.sh` 是一次具体实验配置：模型大小、batch、学习率、数据路径和日志位置都在 shell 中指定。

### 5.2 Full SFT / Agent SFT

`trainer/train_full_sft.py` 仍是交叉熵训练，但数据已经是对话或 Agent 示例。当前工作区对它做了 Agent 数据和 assistant-only mask 适配。

`scripts/run_agent_sft.sh` 从共同基础 checkpoint 训练 `agent_sft`。这个 checkpoint 是所有 RL 算法的共同起点：

```text
                    ┌→ PPO
Base → Agent SFT ───┼→ GRPO
                    ├→ CISPO
                    ├→ DAPO
                    └→ GSPO
```

不能把 GRPO 训练结果再作为 DAPO 起点，否则差异同时包含“算法差异”和“不同初始化差异”。

### 5.3 LoRA、DPO 和蒸馏支线

`trainer/train_lora.py` 是 MiniMind LoRA SFT；它不是当前 GSM8K Stage 1 的必要主线，但为后续参数高效微调提供历史基础。

`trainer/train_dpo.py` 使用 chosen/rejected 回答对直接优化偏好，不需要在线 rollout。`scripts/prepare_dpo_data.py` 构建数据，`scripts/eval_dpo_holdout.py` 比较训练前后 preference margin。DPO 是独立研究支线，不应与 RLVR 主结果混在一起。

`trainer/train_distillation.py` 让 student 同时学习真实标签和 teacher logits。`scripts/prepare_kd_data.py`、`eval_distillation.py`、`run_kd_ablation.sh` 负责数据、评测和 CE-only 对照。蒸馏也是历史支线，可用于改善小模型冷启动，但不是 GRPO-family 算法本身。

## 6. Rollout：RL 数据是模型现场生成的

SFT 使用已有答案；RL 的回答由当前策略现场采样，这个过程叫 rollout。

`trainer/rollout_engine.py` 封装：

1. 输入 prompt；
2. 按 temperature/top-k/top-p 采样；
3. 保存每个生成 token；
4. 保存该 token 在行为策略下的 log-prob；
5. 保存有效 completion mask 和 prompt 长度。

为什么要保存旧 log-prob？训练更新参数后，同一 token 在新策略下的概率改变。策略优化需要计算：

```text
ratio = exp(new_logprob - old_logprob)
```

ratio 大于 1 表示新策略更偏爱这个动作，小于 1 表示新策略降低了它的概率。PPO、CISPO、DAPO 和 GSPO 都以不同方式限制 ratio，避免一步更新过猛。

`tests/test_rollout_probabilities.py` 检查 rollout 时记录的概率能否由训练模型重算出来。如果两者不一致，重要性采样比率就失去意义。

## 7. 多轮 Agent 与奖励

`trainer/train_agent.py` 是整个 Stage 1 的主控制器。它负责：

1. 读取 RL 题目；
2. 每题采样 `num_generations` 条轨迹；
3. 解析 assistant 是否发出合法工具调用；
4. 执行 calculator，追加 observation；
5. 继续生成下一轮，直到最终答案或达到轮数限制；
6. 计算奖励；
7. 打包 token、mask、old log-prob、reward；
8. 调用统一策略 loss；
9. 反向传播、更新、记录指标、保存 checkpoint。

严格 RLVR 奖励通常是最终 0/1，但代码同时记录诊断维度：

- 输出格式是否合法；
- 工具名和参数是否合法；
- 工具是否成功执行；
- 是否覆盖题目所需工具；
- 最终答案是否正确；
- 整条任务是否成功。

这些诊断值不一定都进入最终奖励，却能解释模型失败在哪里。`tests/test_agent_rewards.py` 验证奖励边界，`audit_multiturn_serialization.py` 检查多轮轨迹拼接没有把 observation 当作模型动作训练。

## 8. 五种强化学习方法

核心公式集中在 `trainer/policy_optimization.py`。把同一题采样出的多条回答记为一组，每条轨迹有 reward 和 token log-prob。

### 8.1 PPO：需要 critic 的经典方案

PPO 用 critic 估计状态价值，再通过 GAE 得到每个 token 的 advantage。直觉是：

> 某一步的结果比 critic 预期更好，就增加该动作概率；更差就降低概率。

它使用 clipped ratio 防止策略变化过大，并同时训练：

- actor：生成回答；
- critic：预测价值；
- reference：约束策略不要离初始模型太远；
- 可选 reward model 或确定性奖励。

实现主要在 `trainer/train_ppo.py`。优点是理论和工具成熟；缺点是需要 critic，显存、实现和调参成本更高。对稀疏的 GSM8K 0/1 奖励，小模型 critic 也可能很难学准。

### 8.2 GRPO：同题多答案做组内比较

GRPO 不训练 critic。对同一道题采样多个答案，用组内 reward 均值和标准差标准化：

```text
advantage_i = (reward_i - group_mean) / (group_std + epsilon)
```

正确率高于组平均的轨迹获得正 advantage，低于平均的获得负 advantage。

优点：省掉 critic，适合可验证奖励。关键缺点：如果一组答案全错或全对，reward 方差为 0，所有 advantage 都接近 0，这一组没有学习信号。

旧的 `trainer/train_grpo.py` 是上游文件上的扩展；新主线通过 `trainer/train_agent.py` 调用 `policy_optimization.py` 的统一 GRPO 实现。

### 8.3 CISPO：直接限制重要性权重

CISPO 关注 importance-sampling ratio。它不是简单把最终 loss 裁剪，而是限制用于梯度的 token 级重要性权重，并对裁剪值 stop-gradient。

直觉：

> 可以奖励好动作、惩罚坏动作，但不能因为新旧策略概率差得太大，让少量 token 主导整个更新。

项目按 MiniMax-M1 论文语义实现 token-level 全局均值和上界诊断。主要代码在 `policy_optimization.py`，参数从 `train_agent.py` 传入。

### 8.4 DAPO：针对 GRPO 无效组和长回答

DAPO 在 GRPO 基础上解决几个现实问题：

1. **Clip-Higher**：正 advantage 方向允许不同的上界，避免过早压制有价值动作；
2. **Dynamic Sampling**：如果一组全对或全错，就继续采样，尽量得到有成功也有失败的有效组；
3. **Token-level loss**：按有效 token 归一化，而不是让短轨迹和长轨迹产生不合理权重；
4. **Soft overlong punishment**：接近长度上限时逐步惩罚，而不是只在截断瞬间突然判罚。

动态采样提高学习信号，但 rollout 成本可能显著增加。如果模型几乎永远答错，不断重采样也救不了，需要先增强 SFT 或设计课程学习。

`scripts/inspect_rl_learnability.py` 和 `run_dapo_learnability_diagnostic.sh` 就是在正式训练前测“有效组率”，避免花大量 GPU 时间训练零 advantage。

### 8.5 GSPO：按整条序列计算 ratio

token-level ratio 可能在长序列里非常噪声。GSPO 先把一条回答所有有效 token 的 log-ratio 做长度归一化聚合，再得到 sequence-level ratio，并在序列层裁剪。

直觉：

> 奖励评价的是整道题的完整回答，那么策略更新也可以更多地把回答看成整体，而不是让某一个 token 的极端概率变化决定结果。

项目实现的是轨迹级 advantage + 序列级重要性比率，不是 GSPO-token 变体。代码位于 `policy_optimization.py`。

## 9. 为什么需要 reference model、旧策略和同步

训练时有三个容易混淆的模型概念：

- **当前 actor**：正在更新的模型；
- **behavior/old policy**：产生这一批 rollout 的策略，其 log-prob 必须冻结到这批数据结束；
- **reference model**：通常是 SFT 初始模型，用于 KL 约束，防止策略漂移过远。

一次 rollout 后可能对同一批轨迹做多个 `policy_update_epochs`。在这些 epoch 内 old log-prob 不能跟着 actor 更新，否则 ratio 会不断回到 1，PPO-style 目标就失效。

`train_agent.py` 的 `rollout_sync_interval`、old log-prob 打包和误差审计用于维护这个约束。`max_rollout_logprob_mae` 是保护阈值：如果 rollout 记录概率和训练时重算概率差异过大，应停止训练而不是继续产生无效结果。

## 10. 训练前为什么先做 learnability 检查

RL 不会自动从任何初始模型学会任务。对每个 SFT checkpoint，应先测：

- task accuracy；
- 格式正确率；
- 工具执行成功率；
- 每题多次采样是否有成功也有失败；
- DAPO effective-group rate；
- 平均生成长度和截断率。

可能出现三种情况：

1. 全错：SFT 太弱，RL 没正样本，应加强冷启动；
2. 有对有错：最适合组相对 RL；
3. 几乎全对：任务饱和，应提高难度而不是继续训练。

这就是 `inspect_rl_learnability.py`、`prepare_tool_rlvr_challenge_data.py` 和 readiness/pilot shell 链存在的原因。

## 11. 评测如何保证公平

`scripts/eval_agent_rlvr.py`：

- 加载多个 checkpoint；
- 使用相同 holdout 题目；
- 使用相同 decode seeds、temperature 和生成长度；
- 逐条保存 trajectory；
- 汇总准确率、工具成功率、有效组率等。

`summarize_eval_results.py` 把多 seed 结果聚合并计算相对 baseline 的变化；`summarize_rl_metrics.py` 汇总训练过程指标。

必须分别报告：

```text
SFT gain = Accuracy(SFT) - Accuracy(Base)
RL gain  = Accuracy(RL)  - Accuracy(SFT)
```

只报告 RL checkpoint 的最终准确率无法说明收益究竟来自 SFT 还是 RL。

## 12. Shell 流水线应该怎样理解

仓库里大量 `run_*.sh` 不是新算法，而是“实验编排记录”：把 Python 入口、参数、等待条件、日志和输出路径串起来。

可以按历史阶段阅读：

```text
run_student_pretrain.sh
  → run_kd_pipeline_after_pretrain.sh
  → run_rlvr_readiness_after_kd.sh
  → run_dpo_after_readiness.sh
  → run_rlvr_pilot_after_dpo.sh
  → run_architecture_verification_after_pilot.sh
  → run_rlvr_formal_after_architecture.sh
  → run_finalize_after_formal.sh
```

这条链是过去单卡环境的顺序执行方案。它通过日志成功 marker 和 `screen` 会话等待前置任务。迁移到 8 卡后，不应原样复用，因为：

- 多个阶段可以并发；
- 路径写死为 `/root/minimind`；
- 多任务会写相同文件；
- `run_rlvr_ablation.sh` 内部串行遍历全部实验。

`run_math_dapo_optimization.sh`、`run_tool_corrected_ablation.sh` 等是发现精度、log-prob 或冷启动问题后的修正版具体实验，不是通用调度框架。

## 13. 测试和审计为什么与算法同样重要

强化学习代码即使能跑，也可能悄悄算错。各测试对应的风险是：

| 测试 | 防止的问题 |
|---|---|
| `test_policy_optimization.py` | 公式、裁剪方向或 mask 错误 |
| `test_rollout_probabilities.py` | old/new log-prob 不可比 |
| `test_agent_rewards.py` | 错误答案被奖励或合法答案被拒绝 |
| `test_agent_sft_data.py` | user/tool token 被当作训练目标 |
| `test_gsm8k_pipeline.py` | 数据泄漏、答案解析和 calculator 错误 |
| `test_dpo_objective.py` | chosen/rejected 方向颠倒 |
| `test_distillation.py` | KL 精度或梯度累积错误 |
| `test_kv_cache_consistency.py` | cache 加速改变模型输出 |
| `test_architecture_features.py` | GQA/MoE 形状或路由错误 |

`collect_experiment_evidence.py` 进一步保存 commit、dirty status、pip freeze、GPU、checkpoint hash 和退出码，让结果可追溯。

## 14. 当前项目已经完成到哪里

### 已完成或基本完成

- MiniMind 模型和 Pretrain/SFT/LoRA/DPO/PPO/GRPO 基础链路；
- 统一 GRPO/CISPO/DAPO/GSPO 目标；
- 多轮 Agent rollout 和行为策略概率记录；
- calculator、答案 parser、严格 RLVR reward；
- GSM8K 固定切分、manifest 和 Agent 数据；
- assistant-only SFT mask；
- learnability/readiness、pilot、固定集评测和聚合；
- 算法、奖励、概率、数据和架构测试；
- 本地 Agent SFT checkpoint、评测轨迹和一个 GRPO pilot 产物。

### 当前仍在工作区、尚未正式提交

- `dataset/gsm8k.py`；
- `trainer/math_env.py`；
- GSM8K 下载/准备/评测脚本；
- `test_gsm8k_pipeline.py`；
- `trainer/train_agent.py`、`train_full_sft.py` 等相关修改；
- 数据 manifest、下载的 MiniMind 模型包和本地实验产物。

### 还不能声称已经完成

- 完整的 3-seed × 5-algorithm Stage 1 公平矩阵；
- 8 卡远端复现；
- Qwen3-4B LoRA SFT；
- Qwen3-4B PPO/GRPO；
- Stage 1 与 Stage 2 同协议对比；
- 最终可复现的环境 lock、配置文件和实验调度器。

## 15. 下一步应该按什么顺序做

### 第一阶段：先冻结当前可复现状态

1. 审查并提交当前 dirty worktree；
2. 固定 PyTorch/依赖版本；
3. 把 shell 中的超参数迁移到 YAML；
4. 在远端运行所有测试；
5. 对已有 checkpoint 和数据核对 SHA-256。

### 第二阶段：让 8 卡真正高效工作

新增：

- `scripts/env/remote_4090.sh`：统一环境变量；
- `scripts/run_one_rlvr.sh`：一次只跑一个算法和 seed；
- `scripts/launch_experiment_matrix.py`：调度 8 个独立任务；
- `configs/minimind_64m/*.yaml`：保存所有实验配置；
- 唯一 run ID 和独立输出目录。

先做 3-update smoke，再做小预算 pilot；只有有效组率和 loss 正常才启动正式矩阵。

### 第三阶段：完成 MiniMind Stage 1

从同一个 `agent_sft` 分叉：

```text
PPO / GRPO / CISPO / DAPO / GSPO × seeds 42, 43, 44
```

固定训练题、更新次数或 rollout token budget、评测题和 decode seeds；报告均值、标准差、GPU-hour、rollout token 和失败实验。

### 第四阶段：实现 Qwen3-4B

需要新增：

- `trainer/model_adapter.py`：统一 MiniMind/HF 模型接口；
- `trainer/qwen3_adapter.py`：chat template、thinking/tool 格式；
- `trainer/train_qwen_lora_sft.py`：PEFT LoRA/QLoRA；
- `trainer/train_qwen_rl.py`：Qwen actor/reference 与统一策略 loss；
- `configs/qwen3_4b/*.yaml`；
- Qwen adapter、序列化和分布式 smoke tests。

先证明 Qwen SFT 能正确调用 calculator 和输出可解析答案，再接 RL。不要一开始就做 8 卡大规模 RL。

### 第五阶段：最终报告

最终报告至少包含：

- Base、SFT、每种 RL 的绝对准确率；
- SFT gain 和 RL gain；
- 三训练 seed 均值/标准差；
- 训练与 rollout token；
- GPU-hour 和峰值显存；
- 有效组率、截断率、KL、clip fraction；
- 典型成功/失败轨迹；
- MiniMind 到 Qwen3-4B 的迁移差异；
- 哪些结论只适用于当前实验规模。

## 16. 推荐阅读顺序

初学者不要按目录字母顺序读代码，按执行链阅读：

1. `README.md`、`docs/PROJECT_PLAN.md`、`docs/EXPERIMENT_PROTOCOL.md`；
2. `dataset/gsm8k.py`、三个 GSM8K prepare 脚本；
3. `trainer/math_env.py`；
4. `dataset/lm_dataset.py`；
5. `trainer/train_full_sft.py`；
6. `trainer/rollout_engine.py`；
7. `trainer/policy_optimization.py`；
8. `trainer/train_agent.py`；
9. `scripts/eval_agent_rlvr.py`；
10. 对应 tests；
11. 最后再读各个 `run_*.sh`，把它们当作实验配置和历史记录。

读 `train_agent.py` 时不要试图一次理解全部 1400 多行。按“参数 → 初始化 → rollout → reward → pack batch → loss → optimizer → logging/checkpoint”的顺序定位函数，会清楚得多。
