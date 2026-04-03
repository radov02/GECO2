import argparse
import os
from pathlib import Path

import sys
# Ensure local Deformable-DETR ops (which provide a top-level `modules` package)
# are on the import path so imports like `from modules.ms_deform_attn import ...`
# resolve to the repo copy instead of a conflicting PyPI package named `modules`.
_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir / 'Deformable-DETR' / 'models' / 'ops'))

import cv2
import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import ops
from tqdm import tqdm

from models.counter_infer import build_model
from models.matcher import build_matcher
from utils.arg_parser import get_argparser
from utils.data import IOCFish5kDataset, pad_collate_test
from utils.losses import SetCriterion


@torch.no_grad()
def evaluate(args):
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu,
    )

    state_dict = torch.load(
        os.path.join(args.model_path, f'{args.model_name}.pth'),
        map_location=device,
    )['model']
    state_dict = {k if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    matcher = build_matcher(args)
    criterion = SetCriterion(
        0, matcher, {"loss_giou": args.giou_loss_coef},
        ["bboxes", "ce"], focal_alpha=args.focal_alpha,
    )
    criterion.to(device)
    criterion.eval()

    test_dataset = IOCFish5kDataset(
        args.data_path,
        args.image_size,
        split='test',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        return_ids=True,
        training=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate_test,
    )

    visuals_dir = os.path.join(args.model_path, f'{args.model_name}_visuals')
    os.makedirs(visuals_dir, exist_ok=True)

    ae = torch.tensor(0.0).to(device)
    se = torch.tensor(0.0).to(device)
    per_image_results = []

    for img, bboxes, density_map, ids, gt_bboxes, scaling_factor, padwh in tqdm(test_loader, desc="Test"):
        img = img.to(device)
        bboxes = bboxes.to(device)
        gt_bboxes = gt_bboxes.to(device)

        outputs, ref_points, centerness, outputs_coord, masks = model(img, bboxes)

        num_objects_pred = []
        pred_boxes_batch = []
        for idx in range(img.shape[0]):
            thr = 1 / 0.11

            if len(outputs[idx]['pred_boxes'][-1]) == 0:
                num_objects_pred.append(0)
                pred_boxes_batch.append(torch.empty((0, 4), device=device))
            else:
                v = outputs[idx]["box_v"]
                v_thr = v.max() / thr
                mask = v > v_thr
                keep = ops.nms(
                    outputs[idx]["pred_boxes"][mask],
                    v[mask],
                    0.5,
                )
                boxes = outputs[idx]["pred_boxes"][mask][keep]
                boxes = torch.clamp(boxes, 0, 1)

                # Remove bboxes in padded area
                maxw = (img.shape[-1] - padwh[idx][0]).to(device)
                maxh = (img.shape[-2] - padwh[idx][1]).to(device)
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                valid = (center[:, 0] * img.shape[-2] < maxw) & (center[:, 1] * img.shape[-1] < maxh)
                boxes = boxes[valid]
                num_objects_pred.append(len(boxes))
                pred_boxes_batch.append(boxes.detach().cpu())

        num_objects_gt = density_map.flatten(1).sum(dim=1)
        num_objects_pred_t = torch.tensor(num_objects_pred, dtype=torch.float32)

        ae += torch.abs(num_objects_gt - num_objects_pred_t).sum()
        se += torch.pow(num_objects_gt - num_objects_pred_t, 2).sum()

        for idx in range(img.shape[0]):
            per_image_results.append({
                'image_idx': ids[idx].item(),
                'gt_count': num_objects_gt[idx].item(),
                'pred_count': num_objects_pred[idx],
            })

        # Save visualizations: predicted bboxes and centers (per image in batch)
        for idx in range(img.shape[0]):
            pred_boxes = pred_boxes_batch[idx]

            # Load the original (unpadded, unnormalized) image from disk
            img_id = test_dataset.image_ids[ids[idx].item()]
            orig_img_path = str(test_dataset.data_path / f'{img_id}.jpg')
            vis_img = cv2.imread(orig_img_path)
            if vis_img is None:
                continue
            orig_h, orig_w = vis_img.shape[:2]
            sf = scaling_factor[idx].item()

            # Visual settings (matching bbox_annotations.py)
            DOT_RADIUS = 4
            DOT_COLOR = (0, 0, 255)       # red (BGR)
            BBOX_COLOR = (0, 255, 0)       # green (BGR)
            BBOX_THICKNESS = 2
            TEXT_COLOR = (0, 255, 255)      # yellow (BGR)

            if pred_boxes.numel() > 0:
                # Boxes are normalised (x1,y1,x2,y2) in the 1024x1024 padded image.
                # Convert back to original pixel coordinates.
                img_sz = float(img.shape[-1])  # 1024
                boxes_px = pred_boxes.clone() * img_sz / sf

                for b in boxes_px:
                    x1, y1, x2, y2 = (
                        int(b[0].clamp(0, orig_w)),
                        int(b[1].clamp(0, orig_h)),
                        int(b[2].clamp(0, orig_w)),
                        int(b[3].clamp(0, orig_h)),
                    )
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.circle(vis_img, (cx, cy), DOT_RADIUS, DOT_COLOR, -1)

            label = f"count: {num_objects_pred[idx]}"
            cv2.putText(vis_img, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2)

            out_name = os.path.join(visuals_dir, f"{img_id}_pred.png")
            cv2.imwrite(out_name, vis_img)

    n = len(test_dataset)
    mae = ae.item() / n
    rmse = torch.sqrt(se / n).item()
    nae = sum(
        abs(r['gt_count'] - r['pred_count']) / max(r['gt_count'], 1)
        for r in per_image_results
    ) / n

    print(f"Test MAE:  {mae:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test NAE:  {nae:.3f}")

    # Save results to txt file
    model_path_abs = os.path.abspath(os.path.join(args.model_path, f'{args.model_name}.pth'))
    txt_path = os.path.abspath(os.path.join(args.model_path, f'{args.model_name}_test_results.txt'))
    with open(txt_path, 'w') as f:
        f.write(f"Model path: {model_path_abs}\n")
        f.write(f"Config: backbone={args.backbone}, image_size={args.image_size}, "
                f"num_enc_layers={args.num_enc_layers}, emb_dim={args.emb_dim}, "
                f"num_objects={args.num_objects}, reduction={args.reduction}\n")
        f.write(f"\nSummary:\n")
        f.write(f"  Test images: {n}\n")
        f.write(f"  Test MAE:    {mae:.3f}\n")
        f.write(f"  Test RMSE:   {rmse:.3f}\n")
        f.write(f"  Test NAE:    {nae:.3f}\n")
        f.write(f"\n{'Image':>8s}  {'GT':>6s}  {'Pred':>6s}  {'AE':>8s}\n")
        f.write('-' * 36 + '\n')
        for r in per_image_results:
            f.write(
                f"{r['image_idx']:>8d}  {r['gt_count']:>6.0f}  {r['pred_count']:>6d}  "
                f"{abs(r['gt_count'] - r['pred_count']):>8.1f}\n"
            )
    print(f"Results saved to: {txt_path}")


if __name__ == '__main__':
    _script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser('GECO2-IOCfish-Inference', parents=[get_argparser()])

    parser.set_defaults(
        model_name='GECO2_IOCfish',
        data_path=str(_script_dir / 'IOCfish5kDataset' / 'done'),
        model_path=str(_script_dir / 'models'),
        backbone='SAM',
        reduction=16,
        image_size=1024,
        num_enc_layers=3,
        emb_dim=256,
        num_heads=8,
        kernel_dim=3,
        num_objects=3,
        batch_size=1,
        num_workers=8,
        tiling_p=0.0,
        giou_loss_coef=2,
        cost_class=2,
        cost_bbox=1,
        cost_giou=2,
        focal_alpha=0.25,
    )

    args = parser.parse_args()
    print(args)
    print("model_name:", args.model_name)
    evaluate(args)
