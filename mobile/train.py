import datetime
import os
import time
import warnings
import numpy as np
from PIL import Image

import presets
import torch
import torch.utils.data
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from transforms import get_mixup_cutmix


# --- 1. 自定义 Dataset 类 ---
class ImageNet64NPZ(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.transform = transform
        self.data = []
        self.targets = []
        # 必须定义 classes 属性，否则 main 函数会报错
        self.classes = [str(i) for i in range(1000)]

        if split == 'train':
            print(f"正在加载训练集，路径: {root}")
            for i in range(1, 11):
                file_path = os.path.join(root, f'train_data_batch_{i}.npz')
                if os.path.exists(file_path):
                    entry = np.load(file_path)
                    self.data.append(entry['data'])
                    # 标签减1，将 1-1000 转为 0-999
                    self.targets.extend(entry['labels'] - 1)
                else:
                    print(f"警告: 找不到文件 {file_path}")

            if not self.data:
                raise FileNotFoundError(f"在 {root} 下没找到任何训练 npz 文件！")

            self.data = np.vstack(self.data).reshape(-1, 3, 64, 64).transpose((0, 2, 3, 1))
        else:
            print(f"正在加载验证集，路径: {root}")
            file_path = os.path.join(root, 'val_data.npz')
            if not os.path.exists(file_path):
                # 尝试不带后缀的情况
                file_path = os.path.join(root, 'val_data')

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到验证集文件: {file_path}")

            entry = np.load(file_path)
            self.data = entry['data'].reshape(-1, 3, 64, 64).transpose((0, 2, 3, 1))
            self.targets = entry['labels'] - 1

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, int(target)


# --- 2. 训练与评估函数 (保持原样) ---
def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    metric_logger.synchronize_between_processes()
    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


# --- 3. 数据加载核心逻辑 (已修改) ---
def load_data(traindir, valdir, args):
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    ra_magnitude = getattr(args, "ra_magnitude", None)
    augmix_severity = getattr(args, "augmix_severity", None)

    # 替换原来的 ImageFolder
    dataset = ImageNet64NPZ(
        traindir,
        split='train',
        transform=presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
            ra_magnitude=ra_magnitude,
            augmix_severity=augmix_severity,
            backend=args.backend,
            use_v2=args.use_v2,
        ),
    )

    print("Loading validation data")
    preprocessing = presets.ClassificationPresetEval(
        crop_size=val_crop_size,
        resize_size=val_resize_size,
        interpolation=interpolation,
        backend=args.backend,
        use_v2=args.use_v2,
    )

    dataset_test = ImageNet64NPZ(
        root=valdir,
        split='val',
        transform=preprocessing
    )

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


# --- 4. Main 函数 (已修改路径和类别逻辑) ---
def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # 关键修改：直接指向包含 .npz 文件的文件夹
    train_dir = os.path.join(args.data_path, "npz")
    val_dir = os.path.join(args.data_path, "npz")

    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    num_classes = 1000  # ImageNet64 固定 1000 类
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes, use_v2=args.use_v2
    )
    collate_fn = (lambda batch: mixup_cutmix(*default_collate(batch))) if mixup_cutmix else default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler,
        num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    import sys
    sys.path.append('..')
    import Masternet
    best_arch = "SuperResK1K3K1(3,40,1,3,1)SuperResK1K3K1(40,40,1,3,1)SuperResK1K5K1(40,40,1,4,1)SuperResK1K3K1(40,80,2,4,1)SuperResK1K7K1(80,80,1,3,1)SuperResK1K5K1(80,80,1,3,1)SuperResK1K5K1(80,80,1,4,1)SuperResK1K7K1(80,80,1,3,1)SuperResK1K7K1(80,160,2,4,1)SuperResK1K3K1(160,160,1,3,1)SuperResK1K5K1(160,160,1,6,1)SuperResK1K7K1(160,160,1,6,1)SuperResK1K5K1(160,160,1,3,1)SuperResK1K5K1(160,240,2,4,1)SuperResK1K7K1(240,240,1,4,1)SuperResK1K5K1(240,240,1,3,1)SuperResK1K7K1(240,240,1,4,1)SuperResK1K5K1(240,240,1,4,1)SuperResK1K5K1(240,480,2,3,1)SuperResK1K5K1(480,480,1,4,1)SuperResK1K7K1(480,480,1,6,1)SuperResK1K3K1(480,480,1,6,1)SuperResK1K7K1(480,480,1,4,1)"

    model = Masternet.MasterNet(num_classes=num_classes, plainnet_struct=best_arch, no_create=False, no_reslink=False)
    model.to(device)

    # 优化器、调度器逻辑保持不变...
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    parameters = utils.set_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    lr_scheduler = main_lr_scheduler

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # 训练循环开始
    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, None, scaler)
        lr_scheduler.step()
        evaluate(model, criterion, data_loader_test, device=device)


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="/data/datasets/Imagenet64", type=str, help="dataset path")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="images per gpu")
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")

    # 补齐分布式核心参数
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    # 补齐你命令里用到的所有开关
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--model-ema", action="store_true", help="enable tracking EMA")  # 关键：补上这个
    parser.add_argument("--sync-bn", action="store_true", help="Use sync batch norm")
    parser.add_argument("--distributed", action="store_true")

    # 补齐尺寸相关参数
    parser.add_argument("--val-resize-size", default=64, type=int)
    parser.add_argument("--val-crop-size", default=64, type=int)
    parser.add_argument("--train-crop-size", default=64, type=int)
    parser.add_argument("--interpolation", default="bilinear", type=str)
    parser.add_argument("--backend", default="PIL", type=str.lower)
    parser.add_argument("--use-v2", action="store_true")

    # 补齐 EMA 步长等细节（防止代码内部用到报错）
    parser.add_argument("--model-ema-steps", type=int, default=32)
    parser.add_argument("--model-ema-decay", type=float, default=0.99998)
    parser.add_argument("--lr-warmup-method", default="constant", type=str)
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float)
    parser.add_argument("--lr-min", default=0.0, type=float)
    parser.add_argument("--mixup-alpha", default=0.0, type=float)
    parser.add_argument("--cutmix-alpha", default=0.0, type=float)
    parser.add_argument("--label-smoothing", default=0.0, type=float)
    parser.add_argument("--clip-grad-norm", default=None, type=float)
    parser.add_argument("--ra-sampler", action="store_true")
    parser.add_argument("--ra-reps", default=3, type=int)
    parser.add_argument("--cache-dataset", action="store_true")
    parser.add_argument("--use-deterministic-algorithms", action="store_true")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)