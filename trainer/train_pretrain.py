import os
import sys

# 让这个脚本既可以作为 trainer 包内部模块导入，也可以直接运行。
# 这样后面的相对导入、上层目录导入都能正常工作。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 这个导入主要是为了兼容某些 Windows 环境下 pyarrow / torch 的 DLL 冲突问题。
# 这里看起来没有直接使用 datasets，但导入后可以触发相关兼容逻辑。
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)

import argparse
import time
import warnings

import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler
)

# 屏蔽一些不影响训练结果的 warning，让日志更干净。
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    训练一个 epoch。

    这个函数负责训练过程中的核心逻辑：
    - 从 DataLoader 里取 batch
    - 前向传播
    - 计算损失
    - 反向传播
    - 梯度累积
    - 梯度裁剪
    - 学习率更新
    - 日志打印
    - 保存模型

    注意：
    这个函数并不显式接收 model / optimizer / scaler / args，
    而是直接使用外层 main 里定义的全局变量。
    """
    # 记录这个 epoch 开始的时间，用于估算训练速度和剩余时间。
    start_time = time.time()

    # 记录最后一个处理到的 step，便于 epoch 结束后处理未完成的梯度累积。
    last_step = start_step

    # loader 每次返回一个 batch。
    # 这里的 batch 结构来自 PretrainDataset:
    #   input_ids, labels
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 把数据搬到目标设备上，比如 cuda:0、cuda:1 或 cpu。
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # 记录当前 step，后面判断是否需要补一次优化器更新会用到。
        last_step = step

        # 计算当前 step 的学习率。
        # epoch * iters + step 表示“全局训练进度”。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)

        # PyTorch 优化器支持逐个 param_group 修改学习率。
        # 这里通常只有一个 param_group，但写法更通用。
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 混合精度上下文：
        # - CPU 上没有必要使用 autocast
        # - GPU 上则启用 autocast 以提升速度、降低显存占用
        with autocast_ctx:
            # 前向传播。
            # model(input_ids, labels=labels) 通常会返回一个包含 loss / aux_loss / logits 的结果对象。
            res = model(input_ids, labels=labels)

            # 总损失 = 主损失 + 辅助损失。
            # 主损失来自语言建模目标，aux_loss 可能来自模型内部的额外正则或 MoE 辅助项。
            loss = res.loss + res.aux_loss

            # 梯度累积：
            # 如果 accumulation_steps > 1，就把 loss 除一下，
            # 让多个小 batch 的梯度累积效果接近一个大 batch。
            loss = loss / args.accumulation_steps

        # 使用 GradScaler 对 loss 做缩放后反向传播。
        # 这样在 float16 下更稳定。
        scaler.scale(loss).backward()

        # 每累积 accumulation_steps 个 batch，再更新一次参数。
        if step % args.accumulation_steps == 0 or step == iters:
            # 先反缩放梯度，避免后面的梯度裁剪受缩放影响。
            scaler.unscale_(optimizer)

            remainder = step % args.accumulation_steps
            if step == iters and remainder:
                correction = args.accumulation_steps / remainder
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)

            # 裁剪梯度范数，防止梯度爆炸。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新参数。
            scaler.step(optimizer)

            # 更新 scaler 的内部状态。
            scaler.update()

            # 清空梯度，为下一轮累积做准备。
            optimizer.zero_grad(set_to_none=True)

        # 每隔 log_interval 步打印一次日志，或者跑到最后一个 step 时也打印。
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time

            # 这里乘回 accumulation_steps，是因为前面把 loss 除掉了。
            current_loss = loss.item() * args.accumulation_steps

            # aux_loss 如果不存在，就记成 0。
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0

            # 主 logits loss = 总 loss - aux_loss。
            current_logits_loss = current_loss - current_aux_loss

            # 取当前优化器里的学习率。
            current_lr = optimizer.param_groups[-1]['lr']

            # 估计剩余时间，单位是分钟。
            # 这里是一个粗略估计，不是严格精确值。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, '
                f'logits_loss: {current_logits_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, '
                f'lr: {current_lr:.8f}, '
                f'epoch_time: {eta_min:.1f}min'
            )

            # 如果启用了 wandb / swanlab，就把指标同步上去。
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # 到达保存点时保存模型。
        # 只有主进程才允许保存，避免 DDP 多进程同时写文件。
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()

            # 如果使用 MoE，就在文件名中加 _moe，便于区分模型结构。
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'

            # DDP 包装后，真正的模型在 model.module 里。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model

            # 如果模型被 torch.compile 包过一层，真实对象可能在 _orig_mod 里。
            # getattr(..., default) 这一写法可以兼容没有 _orig_mod 的情况。
            raw_model = getattr(raw_model, '_orig_mod', raw_model)

            # 取出当前模型参数。
            state_dict = raw_model.state_dict()

            # 保存权重时转成 half precision 并放到 CPU，
            # 可以减小文件体积，也更方便后续加载。
            checkpoint_tmp = ckp + '.tmp'
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, checkpoint_tmp)
            os.replace(checkpoint_tmp, ckp)

            # 另存一个“可断点恢复”的完整状态文件：
            # 里面不仅有 model，还包含 optimizer、scaler、epoch、step 等信息。
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='../checkpoints'
            )

            # 切回训练模式。
            model.train()

            # 主动删除临时变量，减少内存占用。
            del state_dict

        # 及时释放本 step 的中间变量，减少内存压力。
        del input_ids, labels, res, loss

if __name__ == "__main__":
    # 下面是训练脚本入口。

    parser = argparse.ArgumentParser(description="MiniMind Pretraining")

    # ===== 保存相关参数 =====
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")

    # ===== 训练超参数 =====
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # ===== 模型结构参数 =====
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")

    # ===== 数据与权重加载参数 =====
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")

    # ===== 实验记录与编译选项 =====
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")

    args = parser.parse_args()
    if args.accumulation_steps < 1:
        parser.error("accumulation_steps must be >= 1")

    # ============================================================
    # 1. 初始化环境和随机种子
    # ============================================================

    # 如果是 DDP 训练，这里会初始化进程组并返回当前进程的 local_rank。
    # 如果不是 DDP，则返回 0。
    local_rank = init_distributed_mode()

    # DDP 模式下，把当前进程绑定到对应 GPU。
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    # 固定随机种子，尽量保证同样配置下的训练结果可复现。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ============================================================
    # 2. 配置目录、模型参数、检查 checkpoint
    # ============================================================

    # 如果保存目录不存在，就创建它。
    os.makedirs(args.save_dir, exist_ok=True)

    # 根据命令行参数构建模型配置对象。
    # 这一步决定了 hidden_size、层数、是否启用 MoE 等结构参数。
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )

    # 如果 from_resume=1，就尝试从 checkpoint 恢复训练状态。
    # 如果没有恢复文件，则返回 None。
    ckp_data = lm_checkpoint(
        lm_config,
        weight=args.save_weight,
        save_dir='../checkpoints'
    ) if args.from_resume == 1 else None

    # ============================================================
    # 3. 设置混合精度
    # ============================================================

    # 判断当前设备类型，用来决定是否启用 autocast。
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 根据参数选择 bfloat16 或 float16。
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # CPU 上不使用 autocast；
    # GPU 上则启用 autocast 来提升速度并降低显存占用。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ============================================================
    # 4. 初始化 wandb / swanlab
    # ============================================================

    # 默认不启用实验记录。
    wandb = None

    # 只有主进程才初始化日志服务，避免重复创建多个 run。
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        # 断点续训时尽量沿用原来的 run id，这样曲线能接着画。
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None

        # 生成一个比较容易辨认的运行名称。
        wandb_run_name = (
            f"MiniMind-Pretrain-"
            f"Epoch-{args.epochs}-"
            f"BatchSize-{args.batch_size}-"
            f"LearningRate-{args.learning_rate}"
        )

        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume=resume
        )

    # ============================================================
    # 5. 构建模型、数据集、优化器
    # ============================================================

    # init_model 会：
    # - 加载 tokenizer
    # - 构建 MiniMindForCausalLM
    # - 如果 from_weight != 'none'，则加载已有权重
    # - 打印参数量
    # - 将模型移动到指定设备
    model, tokenizer = init_model(
        lm_config,
        args.from_weight,
        device=args.device
    )

    # 构造预训练数据集。
    # 每条样本最终会返回 (input_ids, labels)。
    train_ds = PretrainDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len
    )

    # 如果是分布式训练，就用 DistributedSampler 分发数据；
    # 否则就不使用 sampler。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # float16 下启用 GradScaler；
    # bfloat16 一般不需要同样方式的缩放，所以这里只在 float16 时启用。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    # 语言模型常用 AdamW 优化器。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ============================================================
    # 6. 如果存在断点，则恢复模型 / 优化器 / scaler 状态
    # ============================================================

    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型参数
        model.load_state_dict(ckp_data['model'])

        # 恢复优化器状态
        optimizer.load_state_dict(ckp_data['optimizer'])

        # 恢复梯度缩放器状态
        scaler.load_state_dict(ckp_data['scaler'])

        # 恢复训练进度
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ============================================================
    # 7. 可选 torch.compile 与 DDP 包装
    # ============================================================

    # torch.compile 能带来一定加速，但也会增加调试复杂度。
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    # 如果是分布式训练，用 DDP 包装模型。
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ============================================================
    # 8. 开始训练
    # ============================================================

    for epoch in range(start_epoch, args.epochs):
        # 分布式采样器需要在每个 epoch 重新设定 epoch，
        # 这样每个 epoch 的打乱顺序才会变化。
        train_sampler and train_sampler.set_epoch(epoch)

        # 按当前 epoch 重置随机种子。
        # 下面这一步主要用于让每个 epoch 的数据顺序更稳定地变化。
        setup_seed(42 + epoch)

        # 生成一个样本索引列表。
        # torch.randperm 会返回 [0, 1, 2, ..., len(train_ds)-1] 的随机排列。
        indices = torch.randperm(len(train_ds)).tolist()

        # 如果是断点续训，并且不是从 epoch 开头恢复，
        # 那就需要跳过已经训练过的 batch。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0

        # 把基础 sampler 包装成支持“跳过前 N 个 batch”的 sampler。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)

        # 构造 DataLoader。
        # batch_sampler 已经负责 batch 组织，因此这里不再传 batch_size / shuffle。
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )

        if skip > 0:
            Logger(
                f'Epoch [{epoch + 1}/{args.epochs}]: '
                f'跳过前{start_step}个step，从step {start_step + 1}开始'
            )
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ============================================================
    # 9. 清理分布式进程组
    # ============================================================

    # 训练结束后，优雅关闭 DDP 进程组。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
