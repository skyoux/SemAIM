# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import timm
import timm.optim.optim_factory as optim_factory
from timm.utils import ModelEma
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models import models_semaim as models_aim
from engines.engine_pretrain import train_one_epoch
from datasets.datasets import ImagenetLoader
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser('SemAIM pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='saim_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--query_depth', default=12, type=int,
                        help='decoder depth')
    parser.add_argument('--share_weight', action='store_true',
                        help='Share weight between encoder and decoder')

    parser.add_argument('--prediction_head_type', default='MLP', type=str,
                        help='the type of prediction head: MLP or LINEAR')
    parser.add_argument('--gaussian_kernel_size', default=None, type=int,
                        help='Use gaussian blur to smooth the target image')
    parser.add_argument('--gaussian_sigma', default=None, type=int,
                        help='standard deviation of gaussian blur')
    parser.add_argument('--loss_type', default='L2', type=str,
                        help='Calculate loss between prediction and target per pixel: L1 or L2')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')

    # semaim
    parser.add_argument('--permutation_type', default='stochastic', type=str,
                        help='Permutation type for autoregression: zigzag, raster, stochastic, center2out, out2center, saliency,'
                        ' attention, attention_guided, saliency_guided, stochastic_center, attention_center')
    parser.add_argument('--use_ema_model', action='store_true', help='Use ema features as targets for computing loss')
    parser.set_defaults(use_ema_model=False)
    parser.add_argument('--predict_feature', default='none', type=str, help='Use features as targets: none, inference, ema, dino, clip')
    # parser.set_defaults(predict_feature=False)
    parser.add_argument('--attention_type', default='cls', type=str, help='Attention type: gap, cls and self')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=30, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--not_use_fp16', action='store_true', help='whether to use fp16')
    parser.set_defaults(not_use_fp16=False)

    # Dataset parameters
    parser.add_argument('--data_path', default='../imagenet', type=str, help='dataset path')

    parser.add_argument('--output_dir', default='./pretrain/saim_base',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--experiment', default='exp', type=str, help='experiment name (for log)')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    out_dim = 512
    model = models_aim.__dict__[args.model](permutation_type=args.permutation_type,attention_type=args.attention_type,
                                             query_depth=args.query_depth, share_weight=args.share_weight,out_dim=out_dim,
                                             prediction_head_type=args.prediction_head_type,
                                             gaussian_kernel_size=args.gaussian_kernel_size,
                                             gaussian_sigma=args.gaussian_sigma,
                                             loss_type=args.loss_type, predict_feature=args.predict_feature,
                                             norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    if misc.is_main_process():
        print("Model = %s" % str(model_without_ddp))

    # define ema model
    model_ema = None
    teacher_model = None
    if args.use_ema_model:
    # if args.predict_feature == 'ema':
        # assert args.predict_feature == 'ema'
        model_ema = ModelEma(model, decay=0.999, device=args.device, resume='')
    elif args.predict_feature == 'dino':
        teacher_model = timm.models.vit_base_patch16_224(num_classes=0)
        state_dict = torch.load('/path_to_dino_model/dino_vitbase16_pretrain.pth')
        # state_dict = torch.load('/data/code/ssl/checkpoints/ssl_ckpt/ar/ibot_vitbase16_pretrain.pth')
        msg = teacher_model.load_state_dict(state_dict, strict=False)
        print("loaded dino model with msg:", msg)
        teacher_model.to(device)
        teacher_model.eval()
    elif args.predict_feature == 'clip':
        from models.models_clip import build_model
        state_dict = torch.load('/path_to_clip_model/clip_vitbase16_pretrain.pth', map_location='cpu')
        model_clip = build_model(state_dict)
        msg = model_clip.load_state_dict(state_dict, strict=False)
        print("loaded clip model with msg:", msg)
        model_clip.float()
        teacher_model = model_clip.visual
        teacher_model.to(device)
        teacher_model.eval()

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
        find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    if args.not_use_fp16:
        loss_scaler = None
    else:
        loss_scaler = NativeScaler()

    ckpt_path = os.path.join(args.output_dir, f"{args.model}.{args.experiment}.temp.pth")
    if not os.path.isfile(ckpt_path):
        print("Checkpoint not founded in {}, train from random initialization".format(ckpt_path))
    else:
        print("Found checkpoint at {}".format(ckpt_path))
        model_ema_state_dict = model_ema.ema if args.use_ema_model else None
        misc.load_model(args=args, ckpt_path=ckpt_path, model_without_ddp=model, model_ema=model_ema_state_dict,
            optimizer=optimizer, loss_scaler=loss_scaler)

    if global_rank == 0:
        log_dir = os.path.join(args.log_dir, f"{args.model}.{args.experiment}")
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, args.clip_grad,
            log_writer=log_writer,
            args=args, model_ema=model_ema,teacher_model=teacher_model,
        )

        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model": args.model,
        }
        if loss_scaler is not None:
            save_dict['loss_scaler'] = loss_scaler.state_dict()
        if model_ema is not None:
            save_dict['ema_state_dict'] = model_ema.ema.state_dict()

        ckpt_path = os.path.join(args.output_dir, f"{args.model}.{args.experiment}.temp.pth")
        misc.save_on_master(save_dict, ckpt_path)
        print(f"model_path: {ckpt_path}")

        if args.output_dir and ((epoch + 1) % args.saveckp_freq == 0 or epoch + 1 == args.epochs):
            ckpt_path = os.path.join(args.output_dir, "{}.{}.{:04d}.pth".format(args.model, args.experiment, epoch+1))
            misc.save_on_master(save_dict, ckpt_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir,"{}.{}.log.txt".format(args.model,args.experiment)), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
