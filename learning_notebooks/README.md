# MiniMind 本地学习 Notebook

这组 Notebook 用于在 VS Code 中阅读、运行和标注 MiniMind，不以完整训练模型为目标。

建议顺序：

1. `00_环境与项目地图.ipynb`
2. `01_Tokenizer与Token.ipynb`
3. `02_模型结构与前向传播.ipynb`
4. `03_Dataset与Labels.ipynb`
5. `04_单步训练与反向传播.ipynb`
6. `06-minimind预训练.ipynb`
7. `07-微调.ipynb`
8. [08_Agentic_RLVR从零到多轮工具训练.ipynb](./08_Agentic_RLVR从零到多轮工具训练.ipynb)
9. [09_GRPO_CISPO_DAPO_GSPO从公式到代码.ipynb](./09_GRPO_CISPO_DAPO_GSPO从公式到代码.ipynb)
10. [10_简历项目完整工作报告.ipynb](./10_简历项目完整工作报告.ipynb)

使用方法：

1. 用 VS Code 打开 MiniMind 仓库根目录，而不是只打开 `learning_notebooks` 文件夹。
2. 安装 VS Code 推荐的 Python 和 Jupyter 扩展。
3. 打开 Notebook，单击右上角“选择内核”。
4. 选择 `/Users/shelly/miniconda3/envs/minimind/bin/python` 或 `minimind` Conda 环境。
5. 按顺序逐个运行单元格，并在预留的 Markdown 单元格中添加自己的理解。

原则：核心源码仍保留在 `model/`、`dataset/`、`trainer/` 中。Notebook 通过导入、调用和 `inspect.getsource()` 查看真实源码，仅保存实验和注释，避免复制源码后出现版本不一致。

其中 08 与 09 对应新的 RL 学习依赖：先构造可信多轮环境、action mask 和 verifier，再研究四种 policy objective。两个 Notebook 都使用小 tensor 或手工轨迹，不下载模型、不生成可冒充正式训练结果的数字；完整 GPU 命令见 `docs/foundation_model_interview/00_END_TO_END_GUIDE.md`。

10 是简历对齐的总入口：直接读取真实源码和 `out/remote_final/evidence/out` 中的最终证据，逐项核对架构、训练链路、DPO 离线偏好分支、四种在线策略算法与 Agentic RL 结论；完整文字版分析见 `docs/foundation_model_interview/07_RESUME_ALIGNED_TECHNICAL_REPORT.md`。
