# 实验协议

## 公平性规则

1. PPO、GRPO、CISPO、DAPO、GSPO 从同一个 Agent SFT checkpoint 分叉。
2. 所有算法使用相同 GSM8K train/validation/test manifest。
3. 所有算法使用同一个 prompt、calculator、answer parser 和 verifier。
4. test 只在最终评测使用。
5. 至少运行 3 个训练 seed，评测 seed 单独记录。
6. 同时报告 optimizer update、训练 token、rollout token 和 GPU time。

## 模型分支

```text
MiniMind-64M Base
  └── Agent SFT
       ├── PPO
       ├── GRPO
       ├── CISPO
       ├── DAPO
       └── GSPO

Qwen3-4B-Base
  ├── Base（不训练）
  ├── Fresh zero LoRA → Pure GRPO
  └── QLoRA Agent SFT
       ├── SFT only
       └── SFT → GRPO

Qwen3-4B Track 2
  └── Agent-SFT(A)
       ├── GRPO(A compare)
       ├── GRPO(B compare)
       └── Additional Agent-SFT(B)
```

Track 2 的 A/B/test 必须题目级互斥。GRPO(A) 与 GRPO(B) 使用相同 group 数和 rollout 超参数；B compare 与 B SFT 使用同一题目 ID 集合。所有训练分支从同一个 Agent-SFT(A) adapter 独立分叉，不允许串行继承另一分支。

## 结果解释

需要分别报告：

```text
Base → SFT：监督微调收益
SFT → RL：强化学习增量收益
Base → Pure GRPO：无监督冷启动时 RL 的直接收益
Pure GRPO ↔ SFT → GRPO：warm start 对可优化行为和样本效率的影响
```

如果零方差组比例很高，应先报告“模型没有产生有效相对优势”，不能简单得出“RL 方法无效”。

Track 2 还必须报告 A/B Pre-GRPO probe 的 pass@1、pass@8、effective-group rate 和 zero-variance rate；probe 不更新参数，也不能使用官方 test。
