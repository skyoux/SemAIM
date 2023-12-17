# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import torchvision
import cv2
import numpy as np

import util.misc as misc
import util.lr_sched as lr_sched


def generate_saliency(saliency_model, imgs):
    patch_size = 16
    width = imgs.shape[2] // patch_size
    with torch.no_grad():
        # print("imgs:", imgs)
        # d1, d2, d3, d4, d5, d6, d7, d8 = self.saliency_model(imgs)
        d1, _, _, _, _, _, _, _ = saliency_model(imgs)
        saliency = d1[:, 0, :, :] # Bx224x224
        # mx, mn = torch.max(pred), torch.min(pred)
        # pred = (pred - mn) / (mx - mn)
        # print('saliency:',saliency)
        # print("max value:", saliency.max())
        pred = torch.nn.functional.interpolate(saliency.unsqueeze(dim=1), (width, width), 
        mode='bilinear',align_corners=True)
        N, _, _, _ = pred.shape
        pred = pred.reshape(N, -1)
        mx, mn = torch.max(pred, dim=-1, keepdim=True)[0], torch.min(pred, dim=-1, keepdim=True)[0]
        pred = (pred - mn) / (mx - mn + 1e-5)

    return pred, saliency.unsqueeze(dim=1)


def forward_teacher_features(model, x, model_type):
    assert model_type in ['dino', 'clip']
    if model_type == 'dino':
        return forward_features_dino(model, x)
    else:
        return forward_features_clip(model, x)


def forward_features_dino(model, x):
    B = x.shape[0]
    x = model.patch_embed(x)

    cls_tokens = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)

    for blk in model.blocks:
        x = blk(x)

    x = model.norm(x)
    # return x[:, 1:, :]
    return x


def forward_features_clip(model, x):
    x = model.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat(
        [model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
        dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + model.positional_embedding.to(x.dtype)
    x = model.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    # x = model.ln_post(x[:, 0, :])
    x = model.ln_post(x)

    if model.proj is not None:
        x = x @ model.proj

    # return x[:, 1:, :]
    return x

def calculate_similarity(tokens):
    tokens = torch.nn.functional.normalize(tokens, p=2, dim=-1)
    similarity = torch.sum(tokens[:, 0, :].unsqueeze(1) * tokens[:, 1:, :], dim=-1)

    mx, mn = torch.max(similarity, dim=1, keepdim=True)[0], torch.min(similarity, dim=1, keepdim=True)[0]
    similarity = (similarity - mn) / (mx - mn + 1e-6)

    return similarity

def applyColorMap_on_tensor(tensor, images, alpha=0.3, norm=False, inverse=False):
    # tensor: B C H W
    heat_map = []
    tensor = tensor.cpu()
    for i in range(tensor.shape[0]):
        temp_map = tensor[i]
        if norm:
            temp_map = (temp_map - temp_map.min()) / (temp_map.max() - temp_map.min() + 1e-5)
        if inverse:
            temp_map = 1 - temp_map  # 这里不应该 1-的，但是显示的colormap反了，所以这样操作了
        temp_map = np.uint8(255 * temp_map)
        temp_map = cv2.applyColorMap(temp_map, cv2.COLORMAP_JET) # 0-255
        heat_map.append(temp_map)
    heat_map = torch.Tensor(np.array(heat_map)).cuda().permute(0, 3, 1, 2)
    heat_map = torch.clip(heat_map * alpha + images * (1 - alpha), 0, 255)
    return heat_map


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm=None,
                    log_writer=None,
                    args=None, model_ema=None, teacher_model=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 20

    accum_iter = args.accum_iter

    if args.use_ema_model:
        assert model_ema is not None
        if epoch < 100:
            model_ema.decay = 0.999 + epoch / 100 * (0.9999 - 0.999)
        else:
            model_ema.decay = 0.9999

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = len(data_loader) * epoch + data_iter_step
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        enc_tokens, attention = None, None

        # generate attention
        feature_attention, self_attention = None, None
        if args.predict_feature == 'none' and 'attention' in args.permutation_type: # stochastic for debug
            model.eval()
            with torch.no_grad():
                if args.use_ema_model:
                    _, feature_attention, self_attention = model_ema.ema(samples, forward_encoder=True)
                else:
                    enc_tokens, feature_attention, self_attention = model(samples, forward_encoder=True)
            model.train()
            feature_attention, self_attention = feature_attention.detach(), self_attention.detach()
            attention = self_attention if args.attention_type == 'self' else feature_attention

        with torch.cuda.amp.autocast(loss_scaler is not None):
            if args.predict_feature == 'inference':
                model.eval()
                with torch.no_grad():
                    enc_tokens, feature_attention, self_attention = model(samples, forward_encoder=True)
                    enc_tokens = enc_tokens.detach()
                    feature_attention, self_attention = feature_attention.detach(), self_attention.detach()
                model.train()
                attention = self_attention if args.attention_type == 'self' else feature_attention
            elif args.predict_feature == 'ema':
                if enc_tokens == None:
                    with torch.no_grad():
                        enc_tokens, _, _ = model_ema.ema(samples, forward_encoder=True)
                    enc_tokens = enc_tokens.detach()
                attention = self_attention if args.attention_type == 'self' else feature_attention
            elif args.predict_feature == 'dino':
                with torch.no_grad():
                    enc_tokens = forward_teacher_features(teacher_model, samples, 'dino')
                enc_tokens = enc_tokens.detach()
                attention = calculate_similarity(enc_tokens)
                feature_attention = attention
            elif args.predict_feature == 'clip':
                with torch.no_grad():
                    enc_tokens = forward_teacher_features(teacher_model, samples, 'clip')
                enc_tokens = enc_tokens.detach()
                attention = calculate_similarity(enc_tokens)
                feature_attention = attention

            loss, permutation, loss_map = model(samples, enc_tokens, attention)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter

        if loss_scaler is None:
            loss.backward()
            if (data_iter_step + 1) % accum_iter == 0:
                norm = 0
                if max_norm is not None:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            else:
                norm = None
        else:
            norm = loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=max_norm,
                           update_grad=(data_iter_step + 1) % accum_iter == 0)
            fp16_scaler = loss_scaler._scaler.get_scale()

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value, total_norm=norm)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            # epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, it)
            log_writer.add_scalar('lr', lr, it)
            log_writer.add_scalar('grad_norm', norm, it)
            if loss_scaler is not None:
                log_writer.add_scalar('fp16_scaler', fp16_scaler, it)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}