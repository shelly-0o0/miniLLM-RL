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

## SVAMP 派生协议

跨数据集实验使用 `arkilpatel/SVAMP` revision `78e727689e1c1bebfc4be39c446898e8e10b0518`。SVAMP 本身是 1,000 题 challenge set，没有本项目可以直接采用的官方 train/test；因此这里只能声称使用作者 augmented cross-validation folds 派生的项目内协议：

```text
fold 0–3 → 816 题 train
fold 4   → 184 题 frozen holdout
train 前 128 题 → 无更新行为 probe
```

除题目 ID 不重叠外，构造脚本还要求 train/holdout 的 `group_nums` 变体家族不重叠。每道题的 prefix equation 被确定性转换为 infix expression，并由项目共用的安全 calculator 重放；结果与原始答案不一致时构造立即失败。

完整来源哈希、类型计数和 processed 文件哈希见 `dataset/manifests/svamp_agent.json`。报告结果时必须称为“derived holdout”，不能称为 SVAMP 官方 test。
