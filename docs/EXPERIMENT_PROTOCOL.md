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

Qwen3-4B Base
  └── LoRA Agent SFT
       ├── GRPO
       └── PPO
```

## 结果解释

需要分别报告：

```text
Base → SFT：监督微调收益
SFT → RL：强化学习增量收益
```

如果零方差组比例很高，应先报告“模型没有产生有效相对优势”，不能简单得出“RL 方法无效”。
