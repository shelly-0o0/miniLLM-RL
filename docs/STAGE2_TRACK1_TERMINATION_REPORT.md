# Stage 2 Track 1 Pure GRPO 终止报告

> 状态：人工终止并保留负结果
> 终止时间：2026-10-08 01:31:23（Asia/Shanghai）
> 模型：Qwen3-4B-Base，fresh zero-initialized LoRA，seed 42

## 1. 决策

Track 1 的 `pure_grpo` 在消费 5,363/6,726 个候选组后停止。停止前已生成 42,904 条真实多轮 on-policy 轨迹，完成 5,352 次参数更新，覆盖原计划预算的 79.74%。

该进程没有崩溃，也没有出现非有限参数。终止原因是实验问题已经得到足够清晰的负证据：最后 200 组仍没有任何 strict success、合法工具调用或工具执行，八条轨迹持续耗尽 `8×384=3,072` 个 action token。继续消费剩余 1,363 组，预计只会重复同一种不可执行行为。

训练日志、metrics、滚动 checkpoint 和 11 份安全拒绝诊断均保留在实验服务器。没有导出终态 adapter，因此该 checkpoint 不进入正式 test 或四臂结果矩阵。

## 2. 终止时计数

| 项目 | 数值 |
|---|---:|
| 候选组 | 5,363 / 6,726 |
| 完成比例 | 79.735% |
| rollout trajectories | 42,904 |
| optimizer updates | 5,352 |
| KL 安全拒绝 | 11 |
| 最长连续拒绝 | 1 |
| 训练 wall time | 194,575.13 s / 54.05 h |
| 终态 adapter | 未导出 |

## 3. 行为变化

| 指标 | 前 200 组 | 后 200 组 |
|---|---:|---:|
| Shaped reward | -2.3541 | +0.2504 |
| Answer accuracy | 5.375% | 43.3125% |
| Protocol progress | 0.7117 | 1.4676 |
| Strict task accuracy | 0% | 0% |
| Format valid rate | 0% | 0% |
| Tool execution success | 0% | 0% |
| Evidence coverage | 0% | 0% |
| Action tokens/group | 3,070.14 | 3,072.00 |
| KL k3 | 0.03788 | 0.05710 |

课程奖励确实让模型更常算出正确最终数字，并提高了部分协议片段得分；但这些改善没有跨过最关键的离散边界：模型始终没有形成可解析、可执行、可由 observation 支撑的工具动作。因而 shaped reward 上升不能被解释为 Agent 能力形成。

## 4. 安全与数值状态

正常更新组的数值均有限，rollout log-prob 对齐也在阈值内。11 个重尾候选组因组级 KL 超过 10 在 backward 前被拒绝，最大被拒绝组 KL 为 157,931.45；所有拒绝都是孤立事件，最长连续拒绝为 1。

这说明安全门按设计工作，但也说明只看常规组的平均 KL 会掩盖极端 reference-drift 尾部。终止是基于行为失败和计算收益，而不是一次数值崩溃。

## 5. 与 warm-start 路线的关系

同配置的 `SFT→GRPO` 已完成 6,726/6,726 组，并在冻结的 1,319 题 official test shard 上取得：

- strict task accuracy：67.4754%；
- answer accuracy：69.5982%；
- format valid rate：98.8628%；
- tool execution success：99.6209%。

训练后段的 warm-start 分支仍有高密度可执行行为，而 cold-start 分支在 42,904 条轨迹后仍为零工具执行。该结果强烈支持“Agent-SFT 提供了必要的行为可达性”这一机制解释。

但 Pure 没有完成计划预算和统一 official test，Base/SFT-only 也没有形成 Track 1 canonical merge。因此不能把上述差异写成完成的四臂效果量，不能报告 Pure 的终态 test accuracy，也不能声称得到了严格预算配平的 Pure-vs-warm 因果估计。

## 6. 可发布结论

可以声称：

1. 在本项目的 Qwen3-4B-Base、fresh LoRA、`G=8`、shaped curriculum 条件下，cold-start GRPO 在 42,904 条轨迹内没有学会可执行 Agent 协议；
2. dense shaping 能提高答案命中和局部协议进度，但不能保证跨越结构化工具调用边界；
3. warm-start SFT→GRPO 已形成稳定工具行为并取得较高冻结测试成功率；
4. KL 安全门成功隔离了 11 个重尾组，没有把异常更新写入策略。

不能声称：

1. “Pure GRPO 完成后准确率为 0%”；它没有完成，也没有终态 test；
2. “GRPO 无法从零学习 Agent”；结论只适用于本配置、预算和奖励；
3. “SFT→GRPO 的 67.48% 全部由 GRPO 带来”；缺少同一 Track 1 的 SFT-only canonical test 对照。

## 7. 证据

机器可读终止摘要见 [`results/stage2_track1/pure_grpo_termination_summary.json`](../results/stage2_track1/pure_grpo_termination_summary.json)。原始大文件不进入普通 Git：

- metrics SHA-256：`89e0021407cbfc6ec26cb04bf91ca90f72b1f0c4ee4f8ac9039622d32f91c6d0`；
- log SHA-256：`c488307a6a0d2108e55563806e435af2b2131981404ec45b5f442759fd89b0c3`。

滚动 checkpoint 被保留用于审计，但不会恢复训练或导出为正式模型，除非以后建立新的、预先注册的 cold-start 协议。
