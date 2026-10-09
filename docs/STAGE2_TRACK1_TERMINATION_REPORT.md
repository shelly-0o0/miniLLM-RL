# Stage 2 Track 1 Pure GRPO 中断与恢复记录

> 四臂研究问题、训练协议与评测设计见 [Stage 2 Track 1（Stack 1）实验设计报告](STAGE2_TRACK1_EXPERIMENT_REPORT.md)。

> 状态：人工中断快照已归档；随后恢复并完成全部训练预算
> 终止时间：2026-10-08 01:31:23（Asia/Shanghai）
> 恢复时间：2026-10-08 01:48:06（Asia/Shanghai）
> 训练完成时间：2026-10-08 15:37:33（Asia/Shanghai）
> 正式测试与终态审计完成时间：2026-10-09 02:10（Asia/Shanghai）
> 模型：Qwen3-4B-Base，fresh zero-initialized LoRA，seed 42

## 1. 决策

Track 1 的 `pure_grpo` 在消费 5,363/6,726 个候选组后曾人工停止。停止前已生成 42,904 条真实多轮 on-policy 轨迹，完成 5,352 次参数更新，覆盖原计划预算的 79.74%。

该进程没有崩溃，也没有出现非有限参数。终止原因是实验问题已经得到足够清晰的负证据：最后 200 组仍没有任何 strict success、合法工具调用或工具执行，八条轨迹持续耗尽 `8×384=3,072` 个 action token。继续消费剩余 1,363 组，预计只会重复同一种不可执行行为。

训练日志、metrics、滚动 checkpoint 和 11 份安全拒绝诊断均保留在实验服务器。2026-10-08 01:48 按用户要求从持久化的第 5,350 组 checkpoint 恢复；中断后的 5,351–5,363 临时记录已连同原始日志按 SHA-256 归档，活动 metrics 回退到 checkpoint 边界后再继续，避免重复组污染最终审计。

恢复控制器同时启动缺失的 Base、SFT-only 和 SFT→GRPO validation shard，以及 Base、SFT-only full official-test shard。Pure 最终完成 6,726/6,726 groups、53,808 trajectories 和 6,712 updates，导出 terminal adapter；统一 128 题 validation 的 strict/format/tool execution 仍全部为 0，answer accuracy 为 42.1875%。因此本文件第 2–4 节保留的是历史中断快照，终态解释应结合完整实验报告与最终评测。

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

截至中断点，正常更新组的数值均有限，rollout log-prob 对齐也在阈值内。11 个重尾候选组因组级 KL 超过 10 在 backward 前被拒绝。恢复后又在 group 5,718、5,768、6,405 隔离 3 个重尾组，终态共 14 次；所有拒绝都是孤立事件，最长连续拒绝为 1。最大被拒绝组的 action-token mean KL 仍为 157,931.45，单 token 最大值为 485,165,152。

这说明安全门按设计工作，但也说明只看常规组的平均 KL 会掩盖极端 reference-drift 尾部。终止是基于行为失败和计算收益，而不是一次数值崩溃。

## 5. 与 Agent-SFT-init GRPO 路线的关系

同配置的 `SFT→GRPO` 已完成 6,726/6,726 组，并在冻结的 1,319 题 official test shard 上取得：

- strict task accuracy：67.4754%；
- answer accuracy：69.5982%；
- format valid rate：98.8628%；
- tool execution success：99.6209%。

训练后段的 SFT→GRPO 分支仍有高密度可执行行为，而 Base-init Pure GRPO 直到 53,808 条轨迹结束仍为零严格成功；末 200 组工具调用仍为 0。统一 validation 进一步确认 Pure 没有形成工具协议。这里的 Agent-SFT 是 RL 系统的冷启动训练；SFT→GRPO 表示使用该冷启动训练产物初始化后再运行 GRPO。

最终 1,319 题 official test 进一步得到：strict 0%、answer 36.9219%、format/tool execution/evidence 均为 0、平均响应 384 token；终态四臂审计为 `PASS`。因此训练期和 validation 的负证据完整延伸到了冻结测试集，但答案命中明显高于 Base 的 4.2456%，再次证明“数学答案代理能力”和“可执行 Agent 协议”是两件不同的事。

SFT-only 的 1,319 题 official-test strict 为 37.6042%，SFT→GRPO 为 67.4754%，同题增量 +29.871 pp。因而不能把 SFT→GRPO 分支的总分全部归因给 Agent-SFT，也不能全部归因给 GRPO：前者建立协议可达性，后者在该起点上继续提高任务质量。实验仍只有一个 training seed，且 SFT/RL 不等 FLOPs，不能外推为普遍因果定律。

## 6. 中断快照可以说明什么

结合完成后的训练和 validation，可以声称：

1. 在本项目的 Qwen3-4B-Base、fresh LoRA、`G=8`、shaped curriculum 条件下，Base-init GRPO（无 Agent-SFT 初始化）在完整 53,808 条轨迹预算内没有学会可执行 Agent 协议；
2. dense shaping 能提高答案命中和局部协议进度，但不能保证跨越结构化工具调用边界；
3. Agent-SFT-init GRPO（SFT→GRPO）已形成稳定工具行为并取得较高冻结测试成功率；
4. KL 安全门成功隔离了 14 个重尾组，没有把异常更新写入策略；
5. SFT→GRPO 相对 SFT-only 的 official-test strict 增量为 +29.871 pp。

不能声称：

1. “GRPO 无法从零学习 Agent”；结论只适用于本配置、预算和奖励；
2. “SFT→GRPO 的 67.48% 全部由 GRPO 带来”；SFT-only 自身已经达到 37.60%；
3. 单 seed 的题目级显著性等同于跨训练重复的稳定性。

## 7. 证据

机器可读中断/恢复摘要见 [`results/stage2_track1/pure_grpo_termination_summary.json`](../results/stage2_track1/pure_grpo_termination_summary.json)。原始大文件不进入普通 Git：

- metrics SHA-256：`89e0021407cbfc6ec26cb04bf91ca90f72b1f0c4ee4f8ac9039622d32f91c6d0`；
- log SHA-256：`c488307a6a0d2108e55563806e435af2b2131981404ec45b5f442759fd89b0c3`。

上述哈希只固定中断时的原始快照。恢复后的 final metrics SHA-256 为 `4ca06064ee337463648b4d8a359eca46461f6d71d88301769b00e060f6da61b8`；恢复运行使用第 5,350 组 checkpoint，最终训练计数和窗口聚合见 [`results/stage2_track1/training_dynamics.json`](../results/stage2_track1/training_dynamics.json)，四臂统计与终态证据见 [`results/stage2_track1/`](../results/stage2_track1/)。
