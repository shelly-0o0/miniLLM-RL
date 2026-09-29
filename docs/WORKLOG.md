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
