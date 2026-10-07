# 工作记录

## 2026-09-29：项目重构启动

### 背景

项目从历史 MiniMind/RL 实验切换为新的 GSM8K Agentic RL 主线，并计划后续迁移到 Qwen3-4B。

### 已执行操作

1. 检查现有仓库状态、远程仓库和最近提交。
2. 确认仓库根目录为 `minimind/`，远程 `origin` 为个人 GitHub 仓库。
3. 确认现有代码包含 MiniMind SFT、LoRA、Agent RL、PPO 入口，以及 GRPO、CISPO、DAPO、GSPO 统一策略目标。
4. 创建开发分支 `gsm8k-agentic-rl`，主分支未被修改。
5. 扩充 `.gitignore`，忽略本地数据、预训练模型、训练日志和原始实验产物。
6. 新增项目目标和阶段计划：`docs/PROJECT_PLAN.md`。
7. 新增 GSM8K 数据与 benchmark 规范：`docs/DATASET_AND_BENCHMARK.md`。
8. 新增公平实验协议：`docs/EXPERIMENT_PROTOCOL.md`。
9. 新增本工作记录：`docs/WORKLOG.md`。
10. 将已被 Git 跟踪的 `out/` 实验产物移出 Git 索引；本地文件未删除。
11. 创建 `configs/`、`scripts/{download,prepare,train,evaluate}/`、`data/` 和 `results/` 的新主线目录。
12. 执行 `git diff --check` 通过。

### 本次未执行

- 未删除本地 `out/`、checkpoint 或数据文件。
- 未删除旧实验报告。
- 未执行 GitHub push。
- 未修改现有训练算法实现。
- 未安装新的 Python/CUDA 依赖；当前环境没有 `torch`，因此完整测试尚未执行。
- `compileall` 在当前 macOS Python 缓存目录权限限制下未完成，未据此判断代码存在语法错误。

### 下一步

1. 把 GSM8K 下载、切分、manifest 生成脚本加入 `scripts/`。
2. 将 calculator、answer parser 和 reward 从现有 Agent 训练脚本中抽成公共模块。
3. 为 MiniMind 建立统一 adapter，先跑通 GSM8K Agent SFT。
4. 用同一 SFT checkpoint 跑五种 Stage 1 RL。
5. 新增 Qwen3-4B adapter 和 LoRA SFT 入口。
6. 完成统一 benchmark 后再提交正式结果。

## 2026-09-29：GSM8K 数据与验证闭环

### 已完成

1. 新增 `dataset/gsm8k.py`，统一原始数据规范化、固定 seed 切分、泄漏检查、JSONL 与 SHA-256 manifest。
2. 新增安全 calculator 与 GSM8K 数值答案 verifier 公共模块 `trainer/math_env.py`。
3. Agent RL 训练中的 calculator 改为复用公共数学环境，避免训练与评测实现漂移。
4. 新增 GSM8K 下载、准备和预测评测命令行脚本。
5. 新增不依赖 GPU/网络的管线单元测试。

### 仍需真实资源

- 下载完整 GSM8K 后生成正式 manifest。
- 准备 MiniMind-64M checkpoint 并执行 Agent SFT。
- 从同一 SFT checkpoint 分叉运行 PPO、GRPO、CISPO、DAPO、GSPO。
- Qwen3-4B LoRA 阶段需要对应模型权重与 GPU 环境。

## 2026-10-03：Stage 2 Qwen3-4B Agentic RL 搭建

### 决策

1. 固定比较矩阵为 Base、Pure GRPO、SFT only、SFT → GRPO。
2. 使用 Qwen3-4B-Base、NF4 QLoRA 和 PEFT adapter。
3. 不把 Stage 2 简化成单轮格式奖励；继续复用 Stage 1 的 calculator、多轮 observation 和严格 RLVR verifier。
4. Pure GRPO 从 fresh zero LoRA 开始；SFT → GRPO 从同一个 SFT adapter 开始，其余预算和环境保持一致。

### 已执行

1. 新增 Qwen3 模板适配、assistant-only SFT 数据集、QLoRA loader、SFT trainer、显存受限 GRPO objective 与 trainer。
2. 新增四组固定评测、readiness audit、smoke/full YAML 和统一运行脚本。
3. 安装并验证 `peft 0.17.1`、`bitsandbytes 0.48.1` 等 Stage 2 依赖。
4. 下载 Qwen3-4B-Base tokenizer/config，不下载 4B 权重。
5. 发现并修正 Qwen3 tool template 的上下文相关 mask 边界，使用 fast-tokenizer offset 精确选择 assistant action。
6. 真实 tokenizer 审计 128 条：零监督 0，最长 515 tokens；监督文本不含 user/tool observation。
7. 用真实 Qwen tokenizer 完成确定性两轮 calculator 协议审计，严格奖励为 +1。
8. 全量单元测试 50/50 通过；数据四个文件的行数和 SHA-256 与 manifest 一致。
9. readiness 报告写入 `out/run_meta/qwen3_stage2_readiness.json`。
10. 完整搭建、算法、命令、监控与结论边界记录于 `docs/STAGE2_QWEN3_4B_BUILD_LOG.md`。
11. 将 Stage 2 隔离部署到远程 `/root/Mini-RL-stage2`，克隆独立 `mini-rl-stage2` Conda 环境，未改动仍在运行的 Stage 1。
12. 远程 8×RTX 4090D 环境重复通过 50/50 测试与 readiness audit。
13. 通过 `hf-mirror.com` 完成 Qwen3-4B-Base 三个权重分片缓存；Hugging Face 主站在该机器上不可达。
14. 在物理 GPU 5 完成 1-step QLoRA SFT smoke；在物理 GPU 5/6 完成 Pure GRPO 与 SFT → GRPO smoke。
15. 三个 smoke adapter 均成功保存；safetensors 全部有限，Pure GRPO 和 SFT → GRPO 相对初始 adapter 均发生可测参数变化。
16. 将模型加载参数由弃用的 `torch_dtype` 更新为 `dtype`，并在远程物理 GPU 7 离线重载 4-bit Qwen3 三个 shard，兼容性检查通过。

### 截至 2026-10-04 的边界（后续状态见下节）

- 当前本地执行环境报告 `cuda_available=false`；GPU smoke 是在隔离的远程目标机完成，而不是本机。
- 目标 GPU 机器仍需启动完整 SFT、Pure GRPO 和 SFT → GRPO。
- validation/test 结果尚未产生，不能提前给出 Stage 2 效果结论。

## 2026-10-03：Stage 2 Track 2 互斥 A/B 实验搭建

### 决策

1. 官方 7,473 条 train 按 seed 42 一次性分成互斥 A=3,736、B=3,737；官方 test 1,319 条保持只评测。
2. 所有训练分支共享 Agent-SFT(A) 起点，比较 GRPO(A)、GRPO(B) 和 Additional-SFT(B)。
3. A/B GRPO 使用等量 3,686 条可验证比较题；B GRPO 与 B 追加 SFT 的题目 ID 完全一致。
4. 正式 GRPO 前对 A/B 各固定 128 题执行 G=8 无更新 probe，测量 pass@k 和有效组比例。
5. 保留真实 calculator 执行、observation 回填、action mask 和严格 RLVR，不把 Track 2 简化为格式学习。

### 已执行

1. 新增 Track 2 数据准备、manifest、readiness audit、A/B probe、六份 YAML、统一 runner 与定向测试。
2. SFT trainer 支持从现有 adapter 继续训练，以实现 Additional-SFT(B)。
3. SFT、probe、GRPO 增加只用于 smoke 链的 adapter override，并把覆盖写入运行元数据。
4. 增加输出防覆盖；最终 test 配置要求五个模型全部存在。
5. 生成并审计数据：A/B/test 三者交集为 0；A/B mask audit 的 zero-supervision 均为 0。
6. Track 2 定向测试、全仓测试 56/56、readiness audit 和静态语法检查全部通过。
7. 完整设计、算法、命令、监控与结论边界写入 `docs/TRACK2_AB_AGENTIC_RL_BUILD_LOG.md`。
8. 首次远程 GRPO(B) smoke 捕获 `kl_k3=5090.64`；定位为 current forward 单侧启用 LoRA dropout，已改为保留训练模式但关闭 Dropout，并增加每轨迹 KL 更新前硬门禁和回归测试。
9. 同条件复测后第 2 组 KL 降至 0.00803，确认修复有效；另发现 1-step SFT smoke 因调度器学习率为 0 而未改参数，已将 Track 2 SFT smoke 改为 2 steps 并加入参数变化验收。
10. Track 2 隔离部署到远程 `/root/Mini-RL-stage2-track2`，复用环境和模型 cache，但未覆盖 `/root/Mini-RL-stage2` 或干扰正在运行的 Track 1。
11. 远程真实权重完成 Agent-SFT(A)、A/B probe、Additional-SFT(B)、GRPO(A/B) 工程 smoke；自动验收显示 A 有 252/504、B 追加 SFT 有 504/504 个 trainable tensor 改变，全部有限。
12. 正式 A probe 的 1,024 条轨迹全部无合法工具调用；定位为 Qwen 首轮 inference prompt 预填空 thinking block，而 SFT 首轮 tool-call target 没有该 block。
13. 停止并归档旧 probe；实现逐轮 prompt 模式解析和严格 SFT-prefix audit。A/B 各 128 条均验证首轮/工具后模式为 `[true, false]`、mismatch 0，全仓测试增至 58/58。
14. 对齐后 A 池 16×4 短 probe 仍为 0 合法工具调用且平均生成长度等于 384 上限，证明模板漂移不是唯一原因；输出已出现 calculator JSON 内容但缺少工具边界。增加结构 token 加权 SFT：`<tool_call>`、`</tool_call>`、`<|im_end|>` 权重 8，其余 assistant token 权重 1，原 adapter 保留为初始化点。2-step 远程 smoke 中 504/504 个 LoRA tensor 更新且无非有限值。
15. 批判性核对 `jjyaoao/qwen-grpo-gsm8k`：采用 warm start、分层奖励与可观测性思想，不复制其单轮 XML 奖励作为 agent 成功。修复“零工具标签也被算格式有效”的诊断缺陷；probe 现同时报告 strict 与 shaped reward 方差，以实测决定是否进行 shaped-GRPO pilot。全仓回归为 60/60 通过。
16. structure-w8 隐层-only 增量 SFT 完成，但固定 128 条审计显示结构 NLL 仅从 7.7965 降到 7.7566、Top-1 仍为 0，故停止该路线且不运行 rollout。修复 Transformers 4.57 自定义 loss 的 gradient-accumulation 接口，真实加权 loss 约 3.7，旧日志约 59 是 16 倍缩放。
17. 证实 PEFT `all-linear` adapter 只有七类 projection、明确排除 `lm_head`。新增从原 504 tensor 无损扩展到 `lm_head` rank-16 LoRA 的路径，并禁用完整 tied embedding 自动保存。10-step pilot 使结构 NLL 下降 0.1299、`<|im_end|>` Top-1 升至 10.94%，普通 token 无退化，已据此启动完整低秩修复 epoch。
18. 将完整修复验收拆为 adapter 产物、128 条 teacher-forced Top-1/Top-20、2×2 自由生成 smoke 和 16×4 决策 probe 四级门槛。Qwen GRPO 增加可审计的 strict/shaped 模式开关，但正式 A/B 配置继续锁定 strict；shaped 只能在 probe 实测存在组内方差后作为隔离的课程 pilot 使用。
19. 完整 lm-head structure-w8 SFT 完成 231 step、3,692 条 A 轨迹，耗时 1,660 秒，delta-only adapter 151 MB。128 条审计中结构 NLL 7.7965→0.5674、Top-1 0→98.09%、Top-20 0→100%；同时发现普通位置结构标记 Top-1 误报 2.455%、Top-20 候选率 10.538%。
20. 修复 adapter 的 A 池 16×4 probe 得到 strict task accuracy 3.125%、pass@4/effective-group rate 12.5%、合法工具执行 59.375%、format valid 7.8125%、unfinished 0；证明严格成功已可达，但标签过生成和证据覆盖仍是主瓶颈。
21. 修正 shaped reward 对“开闭数平衡但部分不可解析”的零惩罚和 ±3 饱和问题；同一 64 条轨迹离线重评分后，严格成功为 6.0，格式损坏近似成功最高 4.5，16/16 组均保留非零方差。全仓回归增至 61/61。
22. 完成 2-group shaped-GRPO 隔离 pilot：两个组均有非零优势，KL/ratio/logprob 门禁通过，506/506 adapter tensor 更新且全有限；2×2 后验 smoke 未见立即退化。正式 downstream 配置已统一切换到修复 adapter，formal A/B GRPO 仍锁定 strict；远程定向测试 31/31 和 readiness PASS。

### 截至 2026-10-04 的边界（后续状态见正式启动记录）

- 当时尚未启动 Track 2 正式训练或官方 test 评测。
- smoke 的小样本 reward/accuracy 只验证工程链路，不能作为模型效果。
- 当前没有 Track 2 准确率结果，不作效果结论。

## 2026-10-05：Stage 2 Track 1 正式执行与冷启动修复

### 先质疑后验证

1. 没有因参考仓库提到“稀疏 reward”就直接重写训练。先对 Base 做 2×2 rollout，观测到 4/4 轨迹均到 384-token 上限、严格与原 shaped reward 都无组内方差。
2. 检查原始文本后发现 Base 已生成 `calculate_math` 的 JSON schema 和部分正确表达式，只缺少 `<tool_call>...</tool_call>` 协议边界；问题不是完全不会数学，而是环境无法把裸 JSON 当成可执行 action。
3. 新增有上界的 `protocol_progress` 课程信号：识别 JSON schema、工具名、参数合法性、可执行性和结果证据；重复候选只取最大值。裸 JSON 永远不进入环境、不计工具成功、不计严格任务成功，strict reward 仍为 ±1。
4. 对已保存 Base 轨迹离线重评分：严格成功仍为 0；shaped 非零方差组率由 0% 变为 100%，平均组内标准差 0.5375。证明修复提供探索信号但没有伪造 Agent 成功。

### SFT 控制 token 修复

1. 原完整 SFT 已使用全部 6,637 条可靠 Agent oracle，但 adapter 只覆盖七类 projection；教师强制审计显示三个协议 token Top-1/Top-20 都为 0。
2. 从原完整 SFT 无损扩展 `lm_head` rank-16 LoRA，比较结构权重 2 与 4 的 50-step 校准。两者 506/506 tensor 更新且全有限。
3. 50-step 审计均把结构 Top-20 提升到 100%，但 `<tool_call>`/`</tool_call>` Top-1 仍为 0；权重 2 的普通位置结构 Top-20 误触发率为 6.99%，低于权重 4 的 9.67%，因此继续评估权重 2 的 100/200-step 最小充分点，而不是盲目跑完整增量 epoch。

### Rollout 与资源工程

1. 将同一 GRPO prompt 的首轮 G 条生成合并为一个 GPU batch；每条发生工具调用后仍独立续写，保留多轮 observation、逐 token logprob 和 action mask。
2. 修复批量生成时较短序列的 PAD mask，PAD 只作存储填充，不能进入 policy action ledger。
3. Base 2×2 探针由约 110 秒降到 65 秒；16×8 共 128 条 Base 轨迹耗时 501.3 秒。
4. 尝试批量计算 policy/reference 的无梯度校验 logprob；实际无加速，且 4-bit 内核因 batch shape 改变使更新前 ratio 从精确 1 偏到 1.00027，因此撤回该修改，保留逐轨迹校验。
5. 单张 RTX 4090D 同放 policy/reference 的 pilot 峰值约 14.2 GiB、每组 38.35 秒，与双卡 38.90 秒相当，且 ratio=1、KL 和 logprob 门禁通过。因此两个正式分支可各占一张 GPU 并行，不再浪费四张卡。

### Pure GRPO 启动证据

1. Base 16×8 预探针：严格成功、合法工具调用均为 0，平均响应 384 tokens；shaped 16/16 组有方差，平均组内标准差 1.1057，answer accuracy 17.19%。
2. 4-group Pure pilot：每组优势非零，KL `0.00060–0.00140`、rollout logprob MAE `0.00557–0.00881`，506/506 adapter tensor 更新且全有限。
3. 已从 fresh zero-equivalent LoRA 启动正式 Pure GRPO：全部 6,726 个训练 prompt、G=8、shaped 课程 reward；最终评测仍只使用严格可执行工具成功。

### SFT 200-step 候选与正式训练恢复门禁

1. 结构权重 2 的 200-step 增量校准正常结束：从 6,637 条 oracle 构成的数据集采样训练 200 个 optimizer step，`train_loss=0.4900607`，506/506 个可训练 tensor 发生变化，最大绝对变化 `0.0113129`，没有非有限 tensor。这里的 200 step 是在“已完整训练一轮的原 SFT adapter”上修复协议 token，并不声称再次遍历了全部 6,637 条。
2. Pure GRPO 首次运行在第 74 个候选组、optimizer step 之前触发逐轨迹 KL 安全门禁：异常轨迹 `KL_k3=12791.203125`，阈值为 10。此前第 1–73 组常规 KL 约在 `0.0002–0.0038`，因此没有直接放宽阈值或把突发值当成正常训练。
3. 最近完整 checkpoint 位于第 50 组，包含 policy/reference adapter、optimizer、scheduler 及 Python/Torch/CUDA RNG 状态。恢复从这个 checkpoint 重放，不从头训练，也不使用第 51–73 组未 checkpoint 的更新。
4. 新增 token 级 KL 故障诊断：在门禁抛错前保存最大异常 action token、policy/old/reference logprob、k3、轨迹文本、工具 turn 和 stop trace。失败组仍不执行 optimizer step，也不推进正式 checkpoint。
5. 修复恢复过程的 provenance：首次 `used_config.yaml` 与 `runtime_overrides.json` 变为不可覆盖；每次启动/恢复另存 segment 配置并追加 `run_segments.jsonl`。本次恢复前已记录原两文件 SHA-256。
6. 四组评测改为 `require_all_runs: true`；正式输出记录数据、配置和每个 adapter 的 SHA-256，且拒绝覆盖已有 summary。新增终态审计脚本，只有完整 SFT、两条全量 GRPO、四组 validation（以及最终 test）全部存在且 finite/预算/哈希一致时才 PASS。
7. 用同一组 16 个 validation prompt、每题 4 条轨迹对比低权重候选：w2-100 的合法/成功工具执行率为 17.19%，w2-200 为 28.13%；w2-200 的 evidence coverage 为 4.69%，但二者严格任务成功均为 0。它们证明课程 reward 可产生方差，却不足以作为正式 warm start。
8. 只读复核已有 Track 2 正式 w8 证据，而不是凭印象决定：三个协议边界 token 的 teacher-forced Top-1 分别为 99.78%、97.09%、96.88%，普通位置结构 token Top-1 误触发约 2.46%；16×4 自由探针的合法/成功工具执行率为 59.38%，严格任务成功率 3.13%。该结果否定了“结构权重 8 必然过度生成”的假设。
9. 因此 Track 1 正式 SFT 定义为两段式：原 adapter 已用全部 6,637 条 oracle 完整训练一轮；随后扩展 `lm_head` LoRA，以结构权重 8 再完整遍历同一训练集一轮。100/200-step w2 只保留为校准产物，不进入四组最终矩阵。
10. Pure 第 74 组故障被确定性复现。异常只来自重复退化轨迹中的一个 `"iguous"` token：current 与 old logprob 都是 `-18.8000`，reference 是 `-3.3928`，说明 resume/重算完全一致；单 token k3 为 `4,911,822`，轨迹均值为 `12,791.20`。
11. 根因是生成使用 `temperature=0.7, top_k=20, top_p=0.95`，而 GRPO ledger 与 reference KL 重算的是未加温度、未截断的完整 softmax。截断重归一化能抽到原始 policy 下概率极低的 token，使样本分布、importance ratio 和 Monte-Carlo KL 的定义不一致。
12. 没有通过放宽 KL、裁剪日志或跳过失败组掩盖问题。训练入口新增同分布硬门禁，只接受 `temperature=1.0, top_k=0, top_p=1.0`；Pure/SFT→GRPO、probe 和最终评测全部同步。旧 73 组连同 checkpoint、metrics、log 和诊断包完整归档为 `pure_grpo_s42_invalid_offpolicy_t0p7_k20_p095_20261005`，不进入正式结果；Pure 从 fresh LoRA 重新启动。
13. full-softmax 重跑在第 3 组再次触发旧的“逐轨迹均值 KL”门禁。token 诊断显示 current/old 完全一致，单条退化轨迹的均值为 `36.0768`，但这条轨迹只占整组 G=8 动作 token 的一部分。训练目标中的 KL 本来按整组所有 action token 全局平均；用逐轨迹均值门禁会把长短轨迹不等权地放大，和实际 loss 口径不一致。
14. 安全门改为 optimizer step 前一次性计算 `group_action_token_mean(KL_k3)`，仍保留阈值 10、有限性检查、最坏轨迹/token 诊断和梯度裁剪，没有删除安全机制或放宽阈值。新增回归测试构造“最坏轨迹均值大于 10、整组 token 均值小于 10”的样例，验证门禁与 streaming GRPO 的 KL 目标严格同口径。
15. 旧逐轨迹门禁运行连同日志、metrics 和诊断保留式归档为 `pure_grpo_s42_invalid_per_trajectory_kl_gate_20261005`。修复后的正式 Pure 从 fresh LoRA 再次启动；原故障第 3 组的组级 KL 为 `4.510293`，未裁剪梯度范数为 `50.3224`，随后按 `max_grad_norm=1` 裁剪并完成更新；第 4 组 KL 回落至 `0.0007819`，到第 9 组仍连续运行。
16. Base on-policy 预探针完成 16 prompt × G=8：严格 pass@1/pass@8 均为 0，严格有效组率 0，平均响应 384 token；但 shaped reward 的 16/16 组都有方差，平均组内标准差 `0.70250`，说明 Pure 分支只有课程信号、没有伪造的严格成功。
17. 正式 lm-head structure-w8 修复完整训练 415 step、6,637 条，耗时 3,199.65 秒，`train_loss=0.468097`。128 条 teacher-forced 审计中结构 token Top-1 达 `99.6396%`，`<tool_call>`/`</tool_call>` 均为 100%，`<|im_end|>` 为 98.4375%；普通位置结构 token Top-1 误触发仅 `0.0847%`。该 adapter 正式进入 SFT-only 与 SFT→GRPO 两臂。
18. SFT→GRPO 的 4-group 隔离 pilot 平均 strict task accuracy `71.875%`，工具调用/执行/required-tool coverage 均为 100%，证据覆盖 75%；最大 KL `0.0002734`、最大 rollout logprob MAE `0.0052268`，全部有限、无安全拒绝。因此从同一正式 SFT adapter 启动全部 6,726 组的 warm-start 正式训练。
19. Pure full-softmax 运行到第 190 组时出现真正的 reference drift：整组 action-token KL 为 `41.8680`，单 token current/old 为 `-20.3327/-20.3327`、reference 为 `-8.5680`。这不是 ledger 错位；门禁正确地在 backward 前终止。该运行完整归档为 `pure_grpo_s42_invalid_group_kl_drift_g190_20261005`。
20. k3 是 sampled Monte-Carlo KL，稀有 token 可产生重尾离群。正式安全策略改为：阈值仍为 10；超过阈值的有限组不反向传播，单独保存 token 诊断并 checkpoint 已消费位置/RNG；最多 64 个、最多连续 3 个，超限仍硬停；非有限值始终立即硬停。`gradient_accumulation_steps` 必须为 1，避免拒绝一组时丢弃别组已累计梯度。旧失败产物不覆盖，新 Pure 从 fresh LoRA 重启。
21. SFT-only 的 128×8 on-policy validation probe 完成：trajectory task accuracy `34.6680%`、pass@1 `34.375%`、pass@8 `72.65625%`、有效组率 `69.53125%`；工具执行成功率 `99.6094%`、evidence coverage `36.3281%`、平均响应 71.80 token、unfinished 0。相对 Base 的 strict 0% 与 384-token 撞顶，warm start 已显著提高可排序轨迹密度；该 probe 仍不替代最终统一四臂 evaluation。
22. 用已知会在第 190 组触发 KL 重尾的 checkpoint 做确定性安全回放：第 190 组 `group_kl_k3=41.8680496` 被拒绝，optimizer update 保持 189；第 191 组 `group_kl_k3=0.00162286` 正常执行并把 update 推进到 190。最终为 191 candidate、190 update、1 rejection，诊断与 adapter 均保存、无 `safety_failure.json`。因此有界拒绝不是只通过 mock 测试，而是在真实 4B 权重和真实 rollout 上完成了端到端验收。
23. Track 2 GRPO(A/B) 正式训练全部完成：各 3,686 组、29,488 条 rollout、3,686 次更新，均无安全拒绝和非有限指标，终态 adapter 已保存。为满足截止时间，新增可审计单臂 shard 与原子合并器；1 题 Base 真实路径预检通过后，五臂官方 1,319 题 test 已分别在 GPU 0/1/2/4/6 并行启动。合并前强制验证 canonical 配置、数据、adapter、轨迹覆盖和有限性，最终结果仍必须通过 Track 2 fail-closed 审计。
24. Track 2 五臂 official test 全部完成、原子合并且终态审计 PASS。严格成功率依次为 Base `0%`、Agent-SFT(A) `0.910%`、GRPO(A) `2.578%`、GRPO(B) `2.654%`、Additional-SFT(B) `42.532%`。20,000 次同题 paired bootstrap 显示 GRPO(A/B) 相对 Agent-SFT(A) 分别提升 `+1.668/+1.744 pp`，但 B−A 仅 `+0.076 pp`、区间跨 0；Additional-SFT(B) 相对 GRPO(B) 高 `39.879 pp`。新增配对分析脚本和 `docs/STAGE2_TRACK2_FINAL_REPORT.md`，明确单训练 seed、非等算力监督和不可外推到普遍 SFT/RL 排名的限制。

### Track 2 正式训练启动

1. A/B 正式起点保持同一个 `agent_sft_a_lm_head_w8_s42_adapter`。远程重新执行 36 项相关回归与 readiness audit，A/B 各 128 行监督 mask 无空监督，generation prefix 无漂移，G=8、各 3,686 个候选组不变。
2. GRPO(A/B) 同步改为 full-softmax on-policy 采样、单卡 policy/reference 和组级 KL 门。先各运行 4 个隔离 pilot：两者所有数值有限；A/B 最大 KL 分别为 `0.0005200/0.0004606`，最大 rollout logprob MAE 为 `0.01175/0.03138`，均低于 `10/0.1` 门限，且 adapter 正常保存。
3. 四组 pilot 恰好均为严格 reward 零方差，不能据此判定 full run 无信号。此前 128×8 正式 probe 的有效组率为 A `25.78125%`、B `17.96875%`，所以保留 strict reward，不把 pilot 临时换成 shaped 后当正式结果。
4. 通过门禁后，GPU 0/1 分别启动 GRPO(A)/GRPO(B) 全部 3,686 组，GPU 4 启动 Additional-SFT(B) 完整一轮。smoke/pilot 使用独立 tag，正式路径启动前均验证不存在，避免覆盖历史产物。
5. 启动后主动反查奖励密度：A/B 的最初约 16–18 组均为 strict 零方差，而 `25.78125%/17.96875%` 的旧 probe 使用的是截断采样，不能作为 full-softmax 分布的正式依据。没有把“数值小试通过”误写成“奖励足够”；另在 GPU 2/5 启动 A/B 各 128×8 的 on-policy probe，并约定结合正式运行前 64 组决定保留、归档还是改用明确标注的课程路线。
6. on-policy 128×8 probe 完成：A 的 trajectory task accuracy `1.3672%`、pass@8/effective-group `9.375%`；B 分别为 `0.8789%`、`7.03125%`。两者 shaped 非零方差组率均为 100%，但正式 A/B 继续使用 strict reward，避免改变问题定义。
7. Additional-SFT(B) 完成全部 3,686 条、231 step，耗时 1,658.43 秒，`train_loss=0.092773`，zero-supervision 为 0，正式 adapter 已保存。
8. GRPO(A/B) 运行到约 1.3k 组时的只读统计：A 有效组率 `174/1373=12.67%`、累计严格成功轨迹 198；B 为 `121/1316=9.19%`、成功轨迹 143。最大 KL 为 `0.04055/0.01389`，最大 rollout logprob MAE 为 `0.04226/0.04760`，所有数值有限，证明正式 strict 训练不是零信号空转。
9. Additional-SFT(B) 终态结构审计：结构 token Top-1 `99.1071%`，两个 tool-call 边界均 100%，普通位置结构误触发 `0.0280%`；没有灾难性遗忘。B seen-pool 16×4 无更新探针的 trajectory task accuracy 为 `59.375%`、pass@4 `93.75%`、工具执行 100%、evidence coverage `59.375%`。该小样本只作行为验收，不作为官方 test 泛化结论。
10. 新增 `scripts/audit_qwen_track2_results.py` 与 runner `audit_results`：只有两个完整 SFT、A/B 全量 GRPO、五臂官方 test、数据/adapter SHA-256、每组 metrics、finite 检查和 bounded-KL 拒绝计数全部一致才 PASS。远端在 GRPO 尚未终止时实测按预期以“no terminal metrics record” fail-closed，而不是把运行中 checkpoint 当成完整结果。

## 2026-10-05：Stage 1 正式收尾

1. 核验 GRPO、CISPO、DAPO、GSPO × training seeds 42/43/44 共 12 个正式 run；每个 run 均完成 6,726 candidate groups、53,808 rollout trajectories，checkpoint 全部 finite。
2. DAPO seed 43 的 malformed tool-name crash 和 seed 43/44 resume candidate-counter 重置问题没有被掩盖；无效产物保留式归档，两个 seed 都从共同 Agent-SFT fresh 重跑到公平候选预算。
3. 补齐 12 个 checkpoint 的 747 题 validation × decode seeds 42/43/44。统计先合并 decode seed，再跨独立 training seed 计算均值与样本标准差。
4. DAPO 以 `2.9154% ± 0.3847 pp` 的严格任务准确率在 validation 排名第一；相对 Agent-SFT 增量 `+0.9966 pp`，题目级 paired bootstrap 95% CI 为 `[+0.2826, +1.6808] pp`。
5. validation 锁定 DAPO 后，只对 Agent-SFT 和 DAPO 三个训练 checkpoint 执行完整 1,319 题 official test。Agent-SFT 为 `2.1986%`，DAPO 为 `3.3190% ± 0.3047 pp`，增量 `+1.1204 pp`；paired item 95% CI 为 `[+0.5391, +1.7185] pp`。
6. test 的工具执行率已约 99.8%–99.9%，但约 96% 轨迹仍是答案错误；当前瓶颈是文字题到正确算式和证据组合，而不是工具协议格式。
7. 新增 `scripts/analyze_gsm8k_stage1.py`：硬审计预算、轨迹计数、checkpoint finite/变化、两层聚合、paired bootstrap、失败归因和 SHA-256 evidence bundle。
8. 新增 `scripts/run_gsm8k_stage1_final_test.sh` 固化不可覆盖的最终测试参数；完整结论、命令、异常和限制写入 `docs/STAGE1_GSM8K_FINAL_REPORT.md`。
9. 使用可写 Hugging Face cache 重新运行全仓回归，67/67 通过。Stage 1 的四算法正式矩阵、统计分析与最终测试至此完成；PPO 未进入本轮锁定矩阵，不作 PPO 效果结论。

## 2026-10-07：结果多维分析与 Track 1 进程复核

1. 远程只读核对 Track 2 最终审计仍为 `PASS`：GRPO(A/B) 各完成 3,686 groups、29,488 rollouts、3,686 updates，0 次 KL 拒绝；五臂 official test 每臂 1,319 题。
2. 聚合 Track 2 全程 metrics，而不是只看最后一行：A/B 全程零方差组率为 84.75%/87.76%，末 200 组为 82.0%/84.5%；strict trajectory accuracy 从首 200 组的 0.8125%/0.9375% 上升到末 200 组的 2.6875%/2.2500%。
3. 补充行为漏斗：Agent-SFT(A)、GRPO(A)、GRPO(B)、Additional-SFT(B) 的 `Strict/Answer` 分别为 8.0%、17.3%、15.7%、88.1%，`Strict/Evidence` 分别为 17.9%、33.0%、31.0%、96.7%。
4. 补充计算成本：Agent-SFT(A)/Additional-SFT(B) 各约 0.461 h；GRPO(A/B) 分别约 15.208/15.885 h。在当前实现中，在线 RL 约慢 33–35 倍，但这不是严格 FLOPs 配平实验。
5. Stage 1 补充相对效果：DAPO test strict 相对 Agent-SFT 提升 50.96%，evidence coverage 提升 47.94%，平均输出缩短 9.09%；三个训练 seed 的 strict 变异系数约 9.18%。
6. 复核 Track 1：SFT→GRPO 已完整结束 6,726 groups；Pure GRPO 在 2026-10-07 17:58（Asia/Shanghai）运行到 4,615/6,726 groups、4,605 updates，仍有 tmux 和单 GPU Python 进程。
7. Pure GRPO 前 200→末 200 组的 shaped reward 从 -2.342 升到 +0.191、answer accuracy 从 5.38% 升到 41.94%、protocol progress 从 0.708 升到 1.441，但 strict accuracy 仍为 0，末 200 组工具调用仍为 0；课程信号在改善，Agent 行为尚未形成。
8. Pure 全程只有 2 个 group 出现工具调用，10/4,615 个候选组因组级 KL 重尾超过阈值被有界拒绝；均为孤立事件，下一组继续更新，无连续拒绝或非有限值。按当前平均 36.28 s/group 估计仍需约 21.3 h，未包含评测与审计。
9. 将上述效果量、行为漏斗、训练动态、效率、KL、统计边界和跨阶段解释写入 `PROJECT_SUMMARY_REPORT.md` 及两个阶段正式报告。
10. 报告修改后重新运行全仓回归，68/68 通过；`git diff --check` 通过。
