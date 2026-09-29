# GSM8K 数据集与 Benchmark 规范

## 数据来源

主数据集使用 Hugging Face 的 `openai/gsm8k`，配置为 `main`。原始数据包含官方 `train` 和 `test` split；项目不得把官方 test 用于训练、冷启动或调参。

下载入口：<https://huggingface.co/datasets/openai/gsm8k>

## 本项目数据层级

```text
data/raw/                 # 原始 GSM8K，禁止提交
data/processed/           # 清洗和切分后的数据，禁止提交
dataset/manifests/       # 数据来源、数量、哈希，提交
dataset/samples/         # 脱敏小样例，可提交
```

推荐切分：

```text
official train → train / validation
official test  → final test only
```

推荐固定随机种子 `42`，并在 manifest 中记录样本数量与 SHA-256。

## 数据格式

原始记录至少保留：

```json
{
  "id": "gsm8k_train_00000",
  "question": "...",
  "answer": "...",
  "source": "openai/gsm8k",
  "config": "main",
  "split": "train"
}
```

Agent 数据增加工具描述，但不能把 `gold_answer` 放入模型 prompt：

```json
{
  "id": "gsm8k_train_00000",
  "question": "...",
  "gold_answer": "42",
  "tools": ["calculator"],
  "prompt": "Solve the problem and use the calculator tool when needed."
}
```

`gold_answer` 只由 verifier 使用。

## Agent 环境

```text
User → Assistant → calculator tool → Observation → Assistant → final answer
```

calculator 必须使用安全的 AST 白名单执行器，只允许数字、括号和基本算术运算，不能直接使用开放式 Python `eval`。

## Reward 与评测

主实验使用统一的数学答案解析器和 verifier。建议记录：

- final answer accuracy
- pass@k
- tool call rate
- valid tool call rate
- tool execution success
- reward mean/std
- zero-variance group rate
- policy KL
- response length
- rollout token 数和耗时

训练集上的 reward 不能替代固定 test/holdout 上的准确率。DAPO 的 accepted-group 统计也不能直接与普通随机采样组比较。
