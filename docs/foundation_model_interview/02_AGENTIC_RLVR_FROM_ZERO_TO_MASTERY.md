# 第二阶段：Agentic RL / RLVR 从零到可运行代码

本文先解决“训练环境、轨迹和奖励从哪里来”，再进入策略优化算法。没有可信的环境执行和 verifier，GRPO、DAPO、GSPO 只是对错误奖励做得更快。

全文固定采用同一条学习链：

```text
看到什么问题 → 为什么发生 → 选择什么方法 → 数学/数据原理
→ MiniMind 代码如何实现 → 怎样运行 → 看什么指标 → 如何证明改善
```

配套实验：[08_Agentic_RLVR从零到多轮工具训练.ipynb](../../learning_notebooks/08_Agentic_RLVR从零到多轮工具训练.ipynb)。

---

## 0. 学完后应该具备什么能力

读完并完成练习后，应当能够：

1. 区分 SFT、普通 RLHF、RLVR 和 Agentic RL；
2. 从一条 JSONL 样本构造 `assistant → tool → observation → assistant` 多轮轨迹；
3. 明确哪些 token 是 policy action，哪些 token 只能作为环境上下文；
4. 解释并修改格式、工具参数、执行结果和最终答案 verifier；
5. 运行 `train_agent.py`，读懂从 rollout 到 optimizer step 的完整调用链；
6. 定位 reward hacking、全零 advantage、上下文截断、工具循环和评测泄漏；
7. 把 mock 工具替换为具有资源限制和可审计 trace 的真实执行环境；
8. 对结果作出可证伪的说明，而不是只展示一条上升的 Reward 曲线。

建议先掌握 Python、PyTorch 的 tensor/mask、Causal LM next-token prediction 和基础概率。策略梯度公式将在下一份教程从零推导，本篇只使用必要结论。

---

## 1. 为什么必须先构建 Agent/RLVR 环境

### 1.1 观察到的问题

普通 SFT 可以让模型模仿：

```text
用户问题 → 某个工具调用 JSON → 某个工具结果 → 最终答案
```

但它并没有亲自执行工具。训练数据里的调用即使参数错误、工具返回过期、最终答案与返回值矛盾，交叉熵仍可能下降。实际部署时还会出现训练集未覆盖的工具组合和错误分支。

### 1.2 根因

SFT 的监督信号是“这个位置应该复现哪个 token”，不是“这个行动执行后是否完成了任务”。它优化的是数据似然，而不是环境中的任务结果。

### 1.3 提出的方法

把模型放入可执行环境：

```text
用户问题
   ↓
模型生成 action
   ↓
环境解析并执行工具
   ↓
tool observation 回写上下文
   ↓
模型继续行动或输出答案
   ↓
verifier 判定任务是否成功
```

如果奖励由程序、单元测试、数据库状态或确定性规则验证，而不是依赖主观 Reward Model，就属于 RLVR（Reinforcement Learning with Verifiable Rewards）范畴。加入多轮外部动作和 observation 后，就是本项目所说的 Agentic RL/RLVR。

### 1.4 能改善什么

合理预期是提升以下能力：

- 选择正确工具，而不只是模仿固定模板；
- 生成能被真实执行器接受的参数；
- 根据 observation 修正后续答案；
- 在完成任务后停止，而不是无限调用；
- 在未见任务组合上优化最终任务成功率。

“合理预期”不是实测结论。本机已经完成代码、单测、冷启动前后评测，以及数学/组合 Tool-Use 各 4 算法 × 3 训练 seed 的 Formal。Agent SFT 把 readiness 集上的数学 task accuracy 从 0% 提升到 59.375%，基础工具从 7.031% 提升到 100%；后者饱和后改用零冷启动标签重合的组合挑战集。两套 RLVR 在当前 10-update 小预算下都没有改变固定采样任务指标，因此项目只把动态采样的信号恢复与成本量化作为 RL 结论，不声称 RL 准确率提升。

---

## 2. 从最基础的 MDP 语言理解多轮工具调用

一条多轮轨迹写成：

```text
τ = (s₁, a₁, o₁, s₂, a₂, o₂, …, s_T, a_T)
```

- `s_t`：模型在第 `t` 轮看到的完整状态，包括 system、工具 schema、用户问题、历史 action 和 observation；
- `a_t`：模型生成的自然语言、思考内容或 `<tool_call>...</tool_call>`；
- `o_t`：环境执行工具后返回的 observation；
- `πθ(a_t|s_t)`：当前模型在状态下生成 action 的概率；
- `R(τ)`：轨迹完成后的奖励；
- `T`：模型给出最终答案、生成 EOS、达到轮数上限或被安全策略终止。

这里最重要的边界是：

```text
assistant 输出 = 策略的 action
tool observation = 环境产生的数据
```

因此训练 mask 应当是：

| token 来源 | 进入模型上下文 | 进入 policy loss | 保存 behavior log-prob |
|---|---:|---:|---:|
| system / user / tool schema | 是 | 否 | 否 |
| assistant 第 1 轮 action | 是 | 是 | 是 |
| tool observation | 是 | 否 | 否 |
| assistant 第 2 轮 action | 是 | 是 | 是 |
| EOS | 是 | 是 | 是 |
| PAD | 仅用于 batch 对齐 | 否 | 否 |

如果把 observation mask 设为 1，模型会被优化去“生成环境结果”，同时 old/current importance ratio 也失去行为策略含义。反过来，如果删除 EOS，模型无法从 loss 中学习停止决策。

多轮轨迹里可能出现多个 EOS：每个 EOS 只关闭当前 assistant turn，并不关闭整条轨迹。不能沿用单轮代码在“第一个 EOS”之后清零 mask，否则第二次工具调用和最终回答都不会获得策略梯度。本实现以 rollout 时逐 token 记录的 action/observation mask 为准，只做 next-token 的一位 shift。

---

## 3. 数据从哪里进入系统

### 3.1 最小数据格式

真实样例在 `tests/fixtures/agent_rl_smoke.jsonl`：

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are a tool-using assistant.",
      "tools": "[{...calculate_math schema...}]"
    },
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"}
  ],
  "gt": ["4"]
}
```

字段语义：

- `conversations[:-1]` 是 rollout 起点；
- 最后一条 assistant 答案只提供数据构造时的参考，不放进 prompt；
- `tools` 是提供给 chat template 和 verifier 的 JSON Schema；
- `gt` 是最终可验证事实列表，不是训练时直接喂给模型的答案。

### 3.2 代码入口

`dataset/lm_dataset.py::AgentRLDataset` 的关键逻辑是：

```python
messages, tools = self.parse_conversations(sample['conversations'])
return {'messages': messages, 'tools': tools, 'gt': sample['gt']}
```

`parse_conversations` 会提取 system 中的 `tools`，并返回 `messages[:-1]`。如果你的数据最后一条并不是答案，这个约定就会错误删除有效 prompt，因此导入新数据前必须先抽样检查角色顺序。

### 3.3 问题：训练/测试题目泄漏

若同一道题同时出现在训练和测试，最终 Accuracy 只证明记忆。项目提供：

```bash
python scripts/split_agent_dataset.py \
  --input dataset/agent_rl_math.jsonl \
  --train_output dataset/agent_rl_math_train.jsonl \
  --eval_output dataset/agent_rl_math_eval.jsonl \
  --manifest out/run_meta/agent_split_manifest.json \
  --eval_ratio 0.1 \
  --seed 42
```

脚本对规范化 user 问题计算 SHA-256，并让同一问题组只能进入一个 split。它只能阻止精确重复，不能阻止换数字、同模板和语义近重复；正式数据还应加入模板分组、MinHash/embedding 去重和人工污染审计。

---

## 4. 问题一：模型不会生成可执行工具格式

### 4.1 现象

- 缺少 `<tool_call>` 结束标签；
- 输出 Python dict 而不是 JSON；
- 使用不存在的工具名；
- required 参数缺失；
- 一段文本里混入多份无法区分的 JSON。

### 4.2 根因

基础模型首先学习的是自然语言分布。严格 schema 在 token 空间中非常稀疏，随机策略几乎不可能一次生成完整合法结构；如果直接用终局 `+1/-1`，同一 prompt 的所有采样往往都失败，组相对 advantage 全为 0。

### 4.3 方法：Tool SFT 冷启动 + 格式 verifier

先用 SFT 教会协议和基本轨迹，再用 RL 根据真实执行结果优化。SFT 解决“会不会说这种语言”，RL 解决“这个行动是否真正完成任务”。

当前解析入口：

```python
def parse_tool_calls(text):
    for body in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        json.loads(body.strip())
```

reward 阶段另行统计：

```text
format_valid = open_tag_count == close_tag_count == parsed_call_count
```

它能发现标签数量不一致和 JSON 解析失败。

### 4.4 怎样证明改善

SFT 前后固定同一组 prompt 和解码参数，报告：

- `format_valid_rate`；
- JSON parse success；
- schema-valid argument rate；
- 最终任务成功率。

只看到格式率提高不能说明任务能力提高，必须同时看最终答案和执行成功。

---

## 5. 问题二：合法 JSON 不等于合法行动

### 5.1 现象

下面两条都能被 `json.loads` 解析，但不一定可执行：

```json
{"name": "unknown_tool", "arguments": {}}
{"name": "calculate_math", "arguments": {}}
```

### 5.2 方法：白名单、参数验证与执行状态分层

当前代码将验证拆为四层：

1. 工具名是否在样本提供的 schema 中；
2. `CHECK_ARGS[name](arguments)` 是否通过；
3. `execute_tool(name, arguments)` 是否返回成功。
4. 重放的执行结果是否覆盖全部 ground truth。

相应指标分别是：

```text
format_valid_rate
tool_call_valid_rate
tool_execution_success_rate
tool_evidence_coverage_rate
```

拆开记录的原因是诊断路径不同：

- 格式失败：优先补 SFT、grammar-constrained decoding；
- 参数失败：补 schema 数据和参数类型校验；
- 执行失败：检查工具、超时、资源与环境状态；
- 执行成功但答案错：检查 observation 使用和最终 verifier。

### 5.3 当前边界

`CHECK_ARGS` 主要检查必填字段是否存在，并不是完整 JSON Schema validator；例如很多字符串类型没有严格验证。生产实现应使用 JSON Schema/Pydantic，并明确额外字段、枚举、数值范围和版本。

当前 `calculate_math` 已从 `eval` 替换为 AST 白名单解释器，只接受数值常量、显式算术节点和受限幂指数，并限制长度、嵌套深度和结果绝对值。这阻止了对象遍历、import 与资源耗尽表达式，但生产级代码执行仍必须放到独立进程/容器，配置 CPU、内存、墙钟、文件系统、系统调用和网络权限。

---

## 6. 问题三：工具执行了，但模型没有利用 observation

### 6.1 现象

模型可能：

- 调用了正确工具，却忽略返回值继续猜答案；
- 调用任意合法工具后直接输出训练记忆；
- 最终答案与 tool result 矛盾；
- 在工具失败后仍假装成功。

### 6.2 方法：把 observation 回写状态

`trainer/train_agent.py::rollout_single` 每轮按如下顺序执行：

```python
messages.append({"role": "assistant", "content": new_text})
result = execute_tool(name, arguments)
messages.append({"role": "tool", "content": result_json})
```

下一轮重新运行 `apply_chat_template(messages, ...)`，因此模型能看到真实 observation。工具 observation 的 token 被追加到 `response_ids`，但对应 `response_mask=0`、`old_logps=0`；它参与下一轮 forward 的条件上下文，不产生策略梯度。

### 6.3 result-aware verifier 与剩余边界

当前 strict verifier 会收集每个合法调用的重放结果，并要求工具结果覆盖全部 GT。因此“调用 `1+1`，然后猜测 `2+2=4`”不能 task success。其核心等价于：

```text
expected_result = canonicalize(execution_trace.last_result)
answer_result   = canonicalize(parse_final_answer(model_output))
result_consistent = equivalent(answer_result, expected_result)
```

该检查只能证明 execution evidence 和 GT 一致，不能完全证明模型在因果上“阅读了 observation”。对状态型任务还应验证最终数据库/文件/页面状态；真实 Agent 还需要下面的反事实 observation 测试。

### 6.4 怎样证明改善

设计反事实测试：

1. 正常 observation；
2. 替换为错误 observation；
3. 隐藏 observation；
4. 工具返回 error。

如果模型真的利用工具，答案会随 observation 合理变化，并在 error 时进行重试或解释失败。只在正常轨迹上测准确率无法排除“先调用、后猜测”。

---

## 7. 问题四：最终答案 verifier 容易被 Reward Hacking

### 7.1 典型漏洞

最简单的 `if gt in text` 会把 ground truth `4` 错判到 `14`；模型也可能在推理中提到正确答案，最后故意输出错误答案。

### 7.2 当前方法

`_final_answer_region` 按优先级提取：

1. 最后一个 `<answer>...</answer>`；
2. 最后一个 `\boxed{...}`；
3. `final answer:`、`最终答案：` 等 marker；
4. 多行文本的最后一个非空行；
5. 否则使用完整文本。

`validate_gt_in_text` 随后：

- 做 Unicode NFKC 规范化；
- 数值用完整数字 token 解析和 `math.isclose`；
- 单一数值 GT 优先比较答案区最后一个数字；
- ASCII 文本使用词边界；
- 多项 GT 要求全部覆盖。

回归测试明确覆盖：

```text
GT=4, output=14                         → 失败
reasoning mentions 4, final answer=14   → 失败
```

### 7.3 仍然不等于通用数学 verifier

字符串/数值比较无法证明：

- `1/2` 与 `0.5` 等价；
- 多项式化简等价；
- 证明过程正确；
- 代码在隐藏测试上正确。

正式数学任务应使用 SymPy canonicalization、任务官方 judge 或受限代码测试。Verifier 本身也需要单元测试、false-positive/false-negative 样本库和版本号。

### 7.4 怎样证明改善

构造 adversarial verifier suite，至少包括：数字子串、单位错误、多个候选答案、推理正确但最终错误、NaN/Inf、大小写与 Unicode、空答案、超长重复文本。报告 verifier 的误判率，而不是默认规则永远正确。

---

## 8. 问题五：奖励太稀疏，训练没有梯度

### 8.1 现象

对同一 prompt 采样 `G` 条，如果全部失败或全部成功：

```text
reward = [-1, -1, -1, -1]
group-relative advantage = [0, 0, 0, 0]
```

策略梯度为零。基础模型尚未学会工具格式时，全失败尤其常见。

### 8.2 两种方法及适用阶段

项目显式提供：

```text
--reward_mode shaped  课程阶段：格式、执行、答案、长度等稠密信号
--reward_mode strict  正式 RLVR：task_success ? +1 : -1
```

Shaped reward 用于把模型带到“偶尔成功”的能力区间；strict reward 用于算法公平对照，避免不同算法通过格式分而不是任务成功取得更高 Reward。

### 8.3 为什么不能一直依赖 shaping

若格式奖励比正确性更容易获得，模型会学习产生大量漂亮 JSON；若思考长度有奖励，模型会把长度本身当目标。正确流程是：

```text
Tool SFT → 可选短期 shaped curriculum → strict RLVR 主训练/主对照
```

并始终把 `task_accuracy` 作为独立硬指标。

### 8.4 结果如何解释

- shaped Reward 上升、task accuracy 不升：大概率 reward hacking 或权重失衡；
- task accuracy 上升、格式率下降：检查 verifier 是否允许绕过工具；
- strict reward 与 accuracy 同步上升：信号一致，但仍需固定测试集验证泛化。

---

## 9. 问题六：多轮轨迹容易无限循环或错误结束

### 9.1 方法

当前代码使用：

- `max_turns`：硬轮数上界；
- 无 tool call：认为进入最终回答并停止；
- EOS：保留为 action；
- 达到最后一轮仍请求工具：`unfinished=True`；
- repetition penalty 与 soft overlong：降低重复/超长倾向。

`task_success` 要求轨迹已闭合，因此 unfinished 即使文本里出现正确答案也不能成功。

### 9.2 需要记录的指标

- `unfinished_rate`；
- 平均/P95 action token；
- 平均工具调用数和平均轮数；
- 重复调用率；
- environment timeout；
- 每个成功任务的 rollout tokens 和工具成本。

### 9.3 结果改善的判断

响应变短不一定更好：可能模型学会高效停止，也可能过早结束。必须联合看 `task_accuracy`、unfinished、长度分位数和成功轨迹成本。

---

## 10. 问题七：old log-prob、上下文和 action mask 错位

这是 RL 代码最隐蔽、也最容易在面试中被深挖的问题。

### 10.1 old log-prob 是什么

rollout 时生成每个 assistant token 的策略叫 behavior policy `π_old`。训练重算当前策略 `π_θ` 后，需要：

```text
ratio_t = exp(log π_θ(a_t|s_t) - log π_old(a_t|s_t))
```

分子和分母必须对应同一个 token、同一个条件上下文。

### 10.2 为什么 rollout 后左截断是错误的

假设 old log-prob 是在 243-token 完整上下文上算的，训练前把序列左截成 128 tokens。虽然 action token 本身还在，它的条件历史已经改变，current log-prob 不再与 old log-prob 可比，第一次更新前 ratio 就会偏离 1。

当前代码因此禁止静默截断：

```text
A rollout trajectory exceeds --max_total_len (...) → RuntimeError
```

正确处理是增大 `max_total_len`，或从 rollout 之前统一缩短 schema、轮数和生成长度，然后从头重跑该 run。

### 10.3 必须通过的 invariant

第一次 policy forward、尚未 optimizer step 时：

```text
rollout_logprob_mae ≈ 0
rollout_ratio_mean ≈ 1
```

本地最终 smoke test 实际得到 `0.0` 和 `1.0`。这只证明对齐正确，不代表模型能力改善。

另一个真实踩坑是：`tokenizer.decode(sampled_ids)` 再重新 encode 不保证回到同一组 token。远端 readiness 曾在第 419 个 token 触发这一错误并拒绝轨迹。修复后维护两份状态：聊天模板文本只用于解析工具调用和提取新增 observation 后缀；模型下一轮输入、action mask 与 old log-prob 始终沿用行为策略真正采样的原始 token ids。回归审计在当前 tokenizer 中找到了一个 36-token 动作重编码为 38-token 的反例，并验证 token ledger 仍逐 token 保留原动作。

---

## 11. `rollout_single` 的逐步代码逻辑

入口：`trainer/train_agent.py::rollout_single`。

### 11.1 构造本轮状态

```python
context = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=tools,
    open_thinking=open_thinking,
)
```

这一步把角色、工具 schema 和历史记录序列化。训练端和 rollout 服务必须使用同一个 tokenizer/chat template，否则 log-prob 和 action 边界都会错。

### 11.2 采样 action

```python
rollout_result = rollout_engine.rollout(
    prompt_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    num_generations=1,
    max_new_tokens=max_new_tokens,
    temperature=0.8,
)
```

返回 completion ids、文本和 per-token old log-prob。EOS 是策略的停止动作，必须保留；PAD 被生成后端禁止采样，若其他后端仍返回 PAD，轨迹直接拒绝，不能静默删除并继续训练。

### 11.3 解析和执行

```python
calls = parse_tool_calls(new_text)
if not calls:
    break
```

有调用则追加 assistant 消息，逐个执行并追加 `role=tool` observation；无调用则把当前输出视为最终回答。

### 11.4 构造跨轮训练序列

```python
response_ids.extend(action_ids)
response_mask.extend([1] * len(action_ids))
response_old_logps.extend(action_old_logps)

response_ids.extend(observation_delta)
response_mask.extend([0] * len(observation_delta))
response_old_logps.extend([0.0] * len(observation_delta))
```

最终形成一条包含 action 与 observation 的连续 token 序列。模型 forward 能看到全部历史，loss 只选中 action。

这里不能将整个 `messages` 再 tokenize 后替换已有序列。正确实现为：

1. 保留本轮原始 `sampled_action_ids`；
2. 用规范化后的 assistant 文本渲染包含 tool observation 的模板；
3. 在规范化 token 流中验证前缀并只切出 `observation_delta`；
4. 将这个环境后缀追加到原始采样 token ledger；
5. 下一轮直接以该 ledger 作为 `input_ids`。

因此即使 `decode → encode` 不可逆，重要性比率的分子与分母仍指向同一动作和同一历史。

### 11.5 Batch 展开

`rollout_batch` 对每个 prompt 复制采样 `num_generations=G` 次，展平顺序是：

```text
prompt_0/gen_0, prompt_0/gen_1, ... prompt_0/gen_G-1,
prompt_1/gen_0, ...
```

因此 reward 通过 `idx // G` 找回原 prompt，group advantage 也能直接 `view(-1, G)`。

---

## 12. `calculate_rewards` 的逐步代码逻辑

入口：`trainer/train_agent.py::calculate_rewards`。

对每条 completion：

1. 找到对应 prompt、GT、tool schema 和全部轮次输出；
2. 去除 `</think>` 之前的思考区，保留最终可见 answer；
3. 统计标签数量并解析所有 tool call；
4. 校验工具白名单与必填参数；
5. 在当前 deterministic mock 中重放工具，统计 execution success；
6. 验证工具执行结果对 GT 的 evidence coverage；
7. 只对未 unfinished 的最终答案运行 `validate_gt_in_text`；
8. 生成分项指标；
9. 根据 `reward_mode` 返回 shaped 分数或 strict `+1/-1`。

严格成功逻辑可以写成：

```text
task_success =
    final_answer_correct
    AND tags_and_json_valid
    AND (not tool_required OR called_at_least_one_tool)
    AND all_called_tools_valid_and_executable
    AND tool_execution_results_cover_all_ground_truth
    AND not unfinished
```

注意：`format_valid_rate=1` 在没有任何 tool tag 时也可能成立，因为开闭标签和解析调用数都是 0。它表示“结构没有损坏”，不表示“按要求调用了工具”；后者由 `calls_satisfied` 和 task success 约束。

---

## 13. 从轨迹到 optimizer step 的完整调用链

```text
AgentRLDataset.__getitem__
  ↓ messages, tools, gt
DataLoader / collate_fn
  ↓ prompt batch
rl_train_epoch
  ↓ rollout_batch
rollout_single
  ↓ completions, prompt_ids, response_ids,
    response_masks, old_logps, turn_outputs, unfinished
calculate_rewards
  ↓ reward_output + binary task_success
可选 dynamic sampling
  ↓ 只保留部分成功的 prompt group
pack input_ids / action mask / old logps
  ↓
reference model forward → reference logps
current model forward   → current logps
group_relative_advantages
compute_policy_loss
  ↓
backward → grad clip → optimizer.step → scheduler.step
  ↓
rollout_engine.update_policy
  ↓
JSONL metrics + checkpoint
```

三个策略必须区分：

- `π_old`：生成本批轨迹的行为策略；用于 importance ratio 分母；
- `π_θ`：正在被更新的当前策略；
- `π_ref`：冻结的 SFT 参考策略；用于 KL 诊断或惩罚。

`old` 和 `reference` 不是同一个概念。

---

## 14. 从 smoke test 到正式训练

### 14.1 第一步：只测 verifier

```bash
cd /path/to/minimind
python -m unittest tests.test_agent_rewards -v
```

预期覆盖合法工具轨迹、unfinished、数字子串攻击、最终答案优先级和超长轨迹保护。

### 14.2 第二步：CPU 随机小模型打通完整循环

从仓库根目录执行：

```bash
cd trainer
python train_agent.py \
  --from_weight none \
  --save_dir /tmp/minimind_agent_smoke_out \
  --save_weight agent_smoke \
  --data_path ../tests/fixtures/agent_rl_smoke.jsonl \
  --device cpu \
  --hidden_size 64 \
  --num_hidden_layers 1 \
  --epochs 1 \
  --batch_size 2 \
  --num_workers 0 \
  --num_generations 2 \
  --max_seq_len 256 \
  --max_gen_len 2 \
  --max_total_len 512 \
  --max_turns 1 \
  --loss_type grpo \
  --policy_update_epochs 2 \
  --beta 0 \
  --reward_mode strict \
  --require_tool_call_for_success 0 \
  --use_reward_model 0 \
  --thinking_ratio 0 \
  --weight_decay 0 \
  --metrics_path /tmp/minimind_agent_smoke_metrics.jsonl
```

随机模型 Accuracy 没有意义；只检查程序不崩、mask/log-prob invariant、loss 有限和 checkpoint 可保存。

### 14.3 第三步：SFT checkpoint 上的小数据调试

```bash
python train_agent.py \
  --from_weight full_sft \
  --data_path ../dataset/agent_rl_math_train.jsonl \
  --save_weight agent_debug \
  --loss_type grpo \
  --batch_size 2 \
  --num_generations 4 \
  --policy_update_epochs 2 \
  --max_gen_len 128 \
  --max_total_len 1024 \
  --reward_mode strict \
  --use_reward_model 0 \
  --debug_mode \
  --debug_interval 1 \
  --metrics_path ../out/metrics/agent_debug.jsonl
```

先人工检查 20～50 条轨迹，确认 GT、工具调用、observation、最终答案和 success 标签一致，再扩大训练。

### 14.4 第四步：固定集评测

```bash
cd ..
python scripts/eval_agent_rlvr.py \
  --weights agent_debug \
  --reference_weight full_sft \
  --data_path dataset/agent_rl_math_eval.jsonl \
  --seeds 42 \
  --limit 256 \
  --num_generations 1 \
  --thinking_ratio 0 \
  --output_dir out/eval_agent_debug
```

`trajectories.jsonl` 是主要证据；`summary.csv` 只是汇总。

---

## 15. 如何读指标并定位问题

| 现象 | 首先检查 | 可能根因 | 建议动作 |
|---|---|---|---|
| Reward 上升、Accuracy 不升 | reward components | 格式/长度被 hack | 用 strict 模式，抽查 false positive |
| 格式率低 | 原始 completion | SFT 不足、模板不一致 | 补 Tool SFT 或约束解码 |
| 工具合法率低 | name/arguments | schema 学习不足 | 增加参数错误/修复样本 |
| 执行率低但合法率高 | execution trace | 超时、类型或工具异常 | 分离环境错误码 |
| 执行成功但答案错 | observation/answer | 模型没利用结果 | result-aware verifier、反事实评测 |
| unfinished 高 | turn outputs | 循环调用、不会停止 | 停止 SFT、循环惩罚、检查 max_turns |
| 所有组 advantage=0 | success 分布 | 任务全会或全不会 | 调整课程难度、动态采样 |
| 首次 ratio 不为 1 | logp/mask/context | tokenizer、截断、位置错位 | 先修 invariant，停止正式训练 |
| 长度降低且准确率降低 | P95/unfinished | 过度长度惩罚 | 减小 penalty、分桶评测 |
| 测试异常高 | split manifest | 泄漏/模板重复 | 重新分组切分和污染审计 |

---

## 16. 从教学 mock 走向生产 Agent

当前实现适合验证算法，但离生产环境还有明确差距：

### 16.1 执行沙箱

必须提供：

- 独立进程或容器；
- CPU/内存/墙钟限制；
- 只读基础文件系统和临时工作区；
- 默认断网、域名级 allowlist；
- syscall/权限隔离；
- 幂等 key 和副作用确认；
- stdout/stderr 截断与敏感信息过滤。

### 16.2 不可伪造 trace

执行环境返回：

```json
{
  "call_id": "...",
  "tool_version": "...",
  "arguments_hash": "...",
  "started_at": "...",
  "finished_at": "...",
  "exit_status": 0,
  "result_hash": "...",
  "result": {...}
}
```

verifier 消费原 trace，不在 reward 阶段重复执行有副作用的操作。

### 16.3 任务级 verifier

- 数学：符号等价、数值容差、单位；
- 代码：隔离隐藏测试和资源限制；
- 浏览器：DOM/后端最终状态；
- 数据库：事务后状态与约束；
- API：响应 schema、状态码和业务不变量。

### 16.4 训练系统

需要增加 environment worker 池、异步 rollout、版本化 policy、trajectory store、失败重试、去重、policy lag 监控和成本配额。吞吐优化不能破坏 `trajectory → exact behavior policy version` 的对应关系。

---

## 17. 结果改善应该怎样写

问题驱动的合格结论应包含四部分。本项目当前可直接使用的真实版本是：

```text
问题：通用 SFT 在数学 Tool-Use readiness 上 task accuracy=0%，基础 Tool-Use=7.031%，
      严格二值奖励产生大量全错组；基础工具在 Agent SFT 后又出现 100% 饱和。
方法：只用训练集 oracle 做 Agent SFT；构建零冷启动标签重合的组合挑战集，加入可执行环境、
      分层 verifier、精确多轮 token ledger 和 strict RLVR，observation 只作上下文。
证据：两套工具数据共 1,600 行 oracle 重放零失败；共同 Agent SFT 初始化；
      4 算法×3 训练 seed；每任务 2,496 条固定评测轨迹与完整 trace。
结果：Agent SFT readiness 数学/基础工具 task accuracy 达到 59.375%/100%；
      组合任务中 DAPO 将所保留组的零方差率降到 0%，但候选组约为普通算法 10.4 倍；
      在当前 10-update、1e-7 小预算下，四算法固定 task success 均为 5.7292%，
      相对 Agent SFT delta=0，因此不宣称 RL 带来准确率提升。
```

上述数字来自固定 holdout 的真实 GPU 产物，不使用 smoke test、训练 Reward 或论文数字。冷启动集合 100% 饱和后，项目构造了不进入 SFT 标签的组合任务并重新做 oracle 审计，而不是继续报告无梯度的 RL run；`prepare_tool_rlvr_challenge_data.py` 就是这一门禁的实际产物。

---

## 18. 进阶练习路线

### 入门

1. 在 Notebook 中运行合法、非法和 reward-hacking 三类 verifier 样例；
2. 手工画出一条多轮轨迹的 action/observation mask；
3. 修改 GT 为 `4`，验证输出 `14` 不会成功。

### 熟练

1. 新增一个纯函数工具，补 schema、参数校验、mock executor 和单元测试；
2. 新增一个 task-specific verifier；
3. 对 32 条数据运行 strict/shaped 对比，解释两个 Reward 为什么不可横比；
4. 从 `trajectories.jsonl` 统计 false-positive verifier 样本。

### 精通

1. 实现 result-aware consistency verifier；
2. 把一个工具迁移到受限 subprocess/container，并生成 signed trace；
3. 增加 observation counterfactual 评测；
4. 实现 environment worker pool 和 policy version 校验；
5. 在数学与 Tool-Use 两套固定集上分别运行四算法、三 seed，并报告准确率、成本和稳定性。

完成这些练习后，再阅读下一阶段的策略优化教程。此时 GRPO/DAPO/GSPO 的 reward、trajectory、mask、old policy 都已有明确来源。
