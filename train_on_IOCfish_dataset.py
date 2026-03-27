import argparse
import os
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import ops

from models.counter import build_model
from models.matcher import build_matcher
from utils.arg_parser import get_argparser
from utils.data import IOCFish5kDataset
from utils.data import pad_collate
from utils.losses import SetCriterion

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def train(args):
    if 'SLURM_PROCID' in os.environ:
        world_size = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        print("Running on SLURM", world_size, rank, gpu)
    else:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )
    print("init dist: ", dist.is_initialized(), rank, world_size, gpu)

    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu,
        find_unused_parameters=True
    )

    backbone_params = dict()
    non_backbone_params = dict()
    for n, p in model.named_parameters():
        if 'backbone' in n:
            backbone_params[n] = p
        else:
            non_backbone_params[n] = p

    optimizer = torch.optim.AdamW(
        [
            {'params': non_backbone_params.values()},
            {'params': backbone_params.values(), 'lr': args.backbone_lr}
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.25)
    if args.resume_training:
        checkpoint = torch.load(os.path.join(args.model_path, f'{args.model_name_resume_from}.pth'))
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = 0
        best = 10000000000000
    else:
        start_epoch = 0
        best = 10000000000000
    best_metrics = None
    matcher = build_matcher(args)
    criterion = SetCriterion(0, matcher, {"loss_giou": args.giou_loss_coef}, ["bboxes", "ce"],
                             focal_alpha=args.focal_alpha)
    criterion.to(device)

    train_dataset = IOCFish5kDataset(
        args.data_path,
        args.image_size,
        split='train',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot,
        training=True
    )
    val_dataset = IOCFish5kDataset(
        args.data_path,
        args.image_size,
        split='val',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        training=True
    )
    test_dataset = IOCFish5kDataset(
        args.data_path,
        args.image_size,
        split='test',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        training=True
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=DistributedSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=DistributedSampler(val_dataset),
        batch_size=1,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=DistributedSampler(test_dataset),
        batch_size=1,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate
    )

    print(rank)
    model_path_abs = os.path.abspath(os.path.join(args.model_path, f'{args.model_name}.pth'))
    txt_path = os.path.abspath(os.path.join(args.model_path, f'{args.model_name}_metrics.txt'))
    if rank == 0:
        with open(txt_path, 'w') as f:
            f.write(f"Model path: {model_path_abs}\n")
            f.write(f"\n{'Epoch':>6s}  {'TrainLoss':>10s}  {'ValLoss':>8s}  {'TrainMAE':>9s}  {'ValMAE':>7s}  {'ValRMSE':>8s}  {'TestMAE':>8s}  {'TestRMSE':>9s}  Best\n")
            f.write('-' * 88 + '\n')
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if rank == 0:
            start = perf_counter()
        train_loss = torch.tensor(0.0).to(device)
        val_loss = torch.tensor(0.0).to(device)
        val_ae = torch.tensor(0.0).to(device)
        val_rmse = torch.tensor(0.0).to(device)
        train_ae = torch.tensor(0.0).to(device)
        test_loss = torch.tensor(0.0).to(device)
        test_ae = torch.tensor(0.0).to(device)
        test_rmse = torch.tensor(0.0).to(device)

        train_loader.sampler.set_epoch(epoch)
        model.train()
        criterion.train()
        for img, bboxes, density_map, img_name, gt_bboxes in train_loader:
            img = img.to(device)
            bboxes = bboxes.to(device)
            gt_bboxes = gt_bboxes.to(device)

            optimizer.zero_grad()
            outputs, ref_points, centerness, outputs_coord, aux = model(img, bboxes)
            outputs_aux, ref_points_aux, centerness_aux, outputs_coord_aux = aux

            losses = []
            num_objects_pred = []

            nms_bboxes = []
            for idx in range(img.shape[0]):
                target_bboxes = gt_bboxes[idx][torch.logical_not((gt_bboxes[idx] == 0).all(dim=1))] / 1024
                l = criterion(outputs[idx],
                              [{"boxes": target_bboxes, "labels": torch.tensor([0] * target_bboxes.shape[0])}],
                              centerness[idx], ref_points[idx])

                l1 = criterion(outputs_aux[idx],
                               [{"boxes": target_bboxes, "labels": torch.tensor([0] * target_bboxes.shape[0])}],
                               centerness_aux[idx], ref_points_aux[idx])
                alpha = 0
                # if min width or height is less than 10px, calculate loss
                if min((target_bboxes[:, 3] - target_bboxes[:, 1]).mean() * args.image_size,
                       (target_bboxes[:, 2] - target_bboxes[:, 0]).mean() * args.image_size) < 25:
                    alpha = 0.3
                keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8],
                               outputs[idx]['box_v'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8], 0.5)

                boxes = (outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8])[keep]
                nms_bboxes.append(boxes)
                num_objects_pred.append(len(boxes))
                losses.append(l['loss_giou'] + l["loss_ce"] + l["loss_bbox"])
                losses.append(l1['loss_giou'] * alpha + l1["loss_ce"] * alpha + l["loss_bbox"] * alpha)
            num_objects_gt = density_map.flatten(1).sum(dim=1)
            loss = sum(losses)

            loss.backward()

            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_loss += loss
            train_ae += torch.abs(num_objects_gt - torch.tensor(num_objects_pred)).sum()

        criterion.eval()
        model.eval()
        with torch.no_grad():
            for img, bboxes, density_map, img_name, gt_bboxes in val_loader:
                img = img.to(device)
                bboxes = bboxes.to(device)
                gt_bboxes = gt_bboxes.to(device)

                outputs, ref_points, centerness, outputs_coord, aux = model(img, bboxes)
                outputs_aux, ref_points_aux, centerness_aux, outputs_coord_aux = aux
                losses = []

                num_objects_pred = []
                nms_bboxes = []

                for idx in range(img.shape[0]):
                    target_bboxes = gt_bboxes[idx][torch.logical_not((gt_bboxes[idx] == 0).all(dim=1))] / 1024

                    l1 = criterion(outputs_aux[idx],
                                   [{"boxes": target_bboxes, "labels": torch.tensor([0] * target_bboxes.shape[0])}],
                                   centerness_aux[idx], ref_points_aux[idx])
                    alpha = 0
                    # if min width or height is less than 10px, calculate loss
                    if min((target_bboxes[:, 3] - target_bboxes[:, 1]).mean() * args.image_size,
                           (target_bboxes[:, 2] - target_bboxes[:, 0]).mean() * args.image_size) < 25:
                        alpha = 1

                    l = criterion(outputs[idx],
                                  [{"boxes": target_bboxes, "labels": torch.tensor([0] * target_bboxes.shape[0])}],
                                  centerness[idx], ref_points[idx])
                    keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8],
                                   outputs[idx]['box_v'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8], 0.5)

                    boxes = (outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8])[keep]
                    nms_bboxes.append(boxes)
                    num_objects_pred.append(len(boxes))
                    losses.append(l['loss_giou'] + l["loss_ce"])
                    losses.append(l1['loss_giou'] + l1["loss_ce"])
                num_objects_gt = density_map.flatten(1).sum(dim=1)
                loss = sum(losses)

                num_objects_pred = torch.tensor(num_objects_pred)

                val_loss += loss * img.size(0)
                val_ae += torch.abs(
                    num_objects_gt - num_objects_pred
                ).sum()
                val_rmse += torch.pow(
                    num_objects_gt - num_objects_pred, 2
                ).sum()

            for img, bboxes, density_map, img_name, gt_bboxes in test_loader:
                img = img.to(device)
                bboxes = bboxes.to(device)
                gt_bboxes = gt_bboxes.to(device)

                outputs, ref_points, centerness, outputs_coord, aux = model(img, bboxes)

                losses = []

                num_objects_pred = []
                nms_bboxes = []

                for idx in range(img.shape[0]):
                    target_bboxes = gt_bboxes[idx][torch.logical_not((gt_bboxes[idx] == 0).all(dim=1))] / 1024

                    l = criterion(outputs[idx],
                                  [{"boxes": target_bboxes, "labels": torch.tensor([0] * target_bboxes.shape[0])}],
                                  centerness[idx], ref_points[idx])
                    keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8],
                                   outputs[idx]['box_v'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8], 0.5)

                    boxes = (outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8])[keep]
                    nms_bboxes.append(boxes)
                    num_objects_pred.append(len(boxes))
                    losses.append(l['loss_giou'] + l["loss_ce"])
                num_objects_gt = density_map.flatten(1).sum(dim=1)
                loss = sum(losses)

                num_objects_pred = torch.tensor(num_objects_pred)

                test_loss += loss * img.size(0)
                test_ae += torch.abs(
                    num_objects_gt - num_objects_pred
                ).sum()
                test_rmse += torch.pow(
                    num_objects_gt - num_objects_pred, 2
                ).sum()

        dist.all_reduce(train_loss)
        dist.all_reduce(val_loss)
        dist.all_reduce(val_rmse)
        dist.all_reduce(val_ae)
        dist.all_reduce(train_ae)
        dist.all_reduce(test_loss)
        dist.all_reduce(test_rmse)
        dist.all_reduce(test_ae)

        scheduler.step()

        if rank == 0:

            end = perf_counter()
            best_epoch = False

            if val_rmse.item() / len(val_dataset) < best:
                best = val_rmse.item() / len(val_dataset)
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict()
                }
                torch.save(
                    checkpoint,
                    os.path.join(args.model_path, f'{args.model_name}.pth')
                )

                best_epoch = True
                best_metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss.item(),
                    'val_loss': val_loss.item(),
                    'train_mae': train_ae.item() / len(train_dataset),
                    'val_mae': val_ae.item() / len(val_dataset),
                    'val_rmse': torch.sqrt(val_rmse / len(val_dataset)).item(),
                    'test_mae': test_ae.item() / len(test_dataset),
                    'test_rmse': torch.sqrt(test_rmse / len(test_dataset)).item(),
                }

            print(
                f"-----------------------------------------------",
                f"Epoch: {epoch}",
                f"Train loss: {train_loss.item():.3f}",
                f"Val loss: {val_loss.item():.3f}",
                f"Train MAE: {train_ae.item() / len(train_dataset):.3f}",
                f"Val MAE: {val_ae.item() / len(val_dataset):.3f}",
                f"Val RMSE: {torch.sqrt(val_rmse / len(val_dataset)).item():.2f}",
                f"Test MAE: {test_ae.item() / len(test_dataset):.3f}",
                f"Test RMSE: {torch.sqrt(test_rmse / len(test_dataset)).item():.2f}",
                f"Epoch time: {end - start:.3f} seconds",
                'best' if best_epoch else ''
            )
            with open(txt_path, 'a') as f:
                f.write(
                    f"{epoch:>6d}  {train_loss.item():>10.3f}  {val_loss.item():>8.3f}  "
                    f"{train_ae.item() / len(train_dataset):>9.3f}  "
                    f"{val_ae.item() / len(val_dataset):>7.3f}  "
                    f"{torch.sqrt(val_rmse / len(val_dataset)).item():>8.2f}  "
                    f"{test_ae.item() / len(test_dataset):>8.3f}  "
                    f"{torch.sqrt(test_rmse / len(test_dataset)).item():>9.2f}  "
                    f"{'*' if best_epoch else ''}\n"
                )
    dist.destroy_process_group()

    if rank == 0 and best_metrics is not None:
        with open(txt_path, 'a') as f:
            f.write(f"\nBest model — epoch {best_metrics['epoch']}:\n")
            f.write(f"  Train loss: {best_metrics['train_loss']:.3f}\n")
            f.write(f"  Val loss:   {best_metrics['val_loss']:.3f}\n")
            f.write(f"  Train MAE:  {best_metrics['train_mae']:.3f}\n")
            f.write(f"  Val MAE:    {best_metrics['val_mae']:.3f}\n")
            f.write(f"  Val RMSE:   {best_metrics['val_rmse']:.2f}\n")
            f.write(f"  Test MAE:   {best_metrics['test_mae']:.3f}\n")
            f.write(f"  Test RMSE:  {best_metrics['test_rmse']:.2f}\n")
        print(f"Metrics saved to: {txt_path}")


if __name__ == '__main__':
    _script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser('GECO2-IOCfish', parents=[get_argparser()])

    # ── IOCfish-specific defaults (all in one place) ───────────────────────
    # These override the generic argparser defaults; any CLI flag still wins.
    parser.set_defaults(
        model_name='GECO2_IOCfish',
        model_name_resumed='GECO2_IOCfish',
        data_path=str(_script_dir / 'IOCfish5kDataset' / 'done'),
        model_path=str(_script_dir / 'models'),
        # Architecture (must match the checkpoint you load / want to save)
        backbone='SAM',
        reduction=16,
        image_size=1024,
        num_enc_layers=3,
        emb_dim=256,
        num_heads=8,
        kernel_dim=3,
        num_objects=3,
        # Training hyper-parameters
        epochs=10,
        lr=1e-5,
        backbone_lr=0.0,
        lr_drop=50,
        weight_decay=1e-4,
        batch_size=4,
        dropout=0.1,
        num_workers=8,
        max_grad_norm=0.1,
        tiling_p=0.5,
        # Loss coefficients
        giou_loss_coef=2,
        cost_class=2,
        cost_bbox=1,
        cost_giou=2,
        focal_alpha=0.25,
        aux_weight=0.3,
    )

    args = parser.parse_args()
    # Ensure the model output directory exists
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    print(args)
    train(args)
