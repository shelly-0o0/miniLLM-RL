# GSM8K Agentic RL 项目计划

## 项目目标

本项目研究可验证数学环境中的 Agentic RL 后训练方法，分为两个阶段：

1. **Stage 1：MiniMind-64M**
   在原有 MiniMind 轻量模型上复用并验证 PPO、GRPO、CISPO、DAPO、GSPO。
2. **Stage 2：Qwen3-4B**
   使用 Hugging Face Qwen3-4B-Base 与 QLoRA，比较 Base、Pure GRPO、SFT only、SFT → GRPO，专门测量 Agent SFT 冷启动对 RL 的影响。
3. **Stage 2 Track 2：互斥 A/B 数据实验**
   将官方 train 一次性分成 A/B；共享 Agent-SFT(A) 起点，比较 GRPO(A)、GRPO(B) 与 Additional-SFT(B)，测量数据新颖性和在线 RL 相对继续监督学习的作用。

项目不以 GSM8K SOTA 为目标，重点是验证训练闭环、奖励信号、方法稳定性和跨模型迁移能力。

## 统一实验链路

```text
GSM8K 原始数据
  → 固定 train/valid/test manifest
  → Agent prompt 与 calculator 环境
  → 统一 answer parser / verifier
  → Agent SFT
  → PPO 或 GRPO-family RL
  → 固定 holdout 评测
```

## 模型与算法矩阵

| 阶段 | 模型 | SFT | RL |
|---|---|---|---|
| Stage 1 | MiniMind-64M | Agent SFT | PPO、GRPO、CISPO、DAPO、GSPO |
| Stage 2 | Qwen3-4B-Base | QLoRA Agent SFT | Pure GRPO、SFT → GRPO |
| Stage 2 Track 2 | Qwen3-4B-Base | Agent-SFT(A)、Additional-SFT(B) | GRPO(A)、GRPO(B) |

Stage 1 的 RL 方法必须从同一个 SFT checkpoint 分叉，不能串行继承前一个算法的 checkpoint。Stage 2 有意保留两个初始化条件：Pure GRPO 使用 Base 上新建的零初始化 LoRA，SFT → GRPO 使用同一个 SFT adapter；二者除此以外共享环境、预算与超参数。

## 研究问题

1. 统一 GSM8K verifier 能否在 64M 模型上产生可学习的 RL 信号？
2. PPO 与 GRPO-family 在稀疏数学奖励下的稳定性有何差异？
3. DAPO 的动态采样能否降低全错/全对组造成的零优势问题？
4. 在相同数据、奖励和评测协议下，方法能否迁移到 Qwen3-4B？
5. RL 相对于 SFT 的增量收益是否值得额外 rollout 成本？
6. 共享 Agent-SFT(A) 起点后，GRPO 在已见 A 与新 B 上的收益是否不同？
7. 在同一 B 题上，在线严格奖励优化与追加 Agent-SFT 的泛化差异是什么？

## 结论边界

结果需要同时报告绝对准确率和相对增益：

```text
SFT gain = Accuracy(SFT) - Accuracy(Base)
RL gain  = Accuracy(RL) - Accuracy(SFT)
```

由于模型大小、训练步数和 rollout 预算有限，结果主要用于说明方法可行性和迁移趋势，不应表述为大模型 SOTA 或普遍能力提升。
