"""Training script for Subject-Aware Sparceiver on HCP1200 task-fMRI.

Usage:
    python train.py --data_dir ./data/sample_data --output_dir ./results
    python train.py --data_dir ./data/sample_data --num_epochs 50 --topk 32
"""

import os
import sys
import time
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from models import SubjectAwareSparceiver
from data import HCPDataset, fmri_collate_fn
from utils import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sparceiver on HCP fMRI')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data/sample_data',
                        help='Root directory of preprocessed trial .npy files')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory for checkpoints and results CSV')

    # Model
    parser.add_argument('--num_latents', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--depth', type=int, default=1, help='Cross-attention iterations')
    parser.add_argument('--self_per_cross_attn', type=int, default=2)
    parser.add_argument('--latent_heads', type=int, default=1)
    parser.add_argument('--cross_heads', type=int, default=1)
    parser.add_argument('--weight_tie_layers', action='store_true', default=True)
    parser.add_argument('--topk', type=int, default=16, help='Sparse attention top-k')
    parser.add_argument('--maxprop', type=float, default=None,
                        help='Sparse attention max-proportion threshold')
    parser.add_argument('--num_freq_bands', type=int, default=6)
    parser.add_argument('--max_freq', type=float, default=10.)

    # Subject-aware
    parser.add_argument('--subj_extraction_layer', type=int, default=0)
    parser.add_argument('--subj_extraction_block', type=int, default=1)
    parser.add_argument('--lambda_subj', type=float, default=1.0,
                        help='Weight for subject prediction loss')

    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_folds', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--warmup_pct', type=float, default=0.15)
    parser.add_argument('--div_factor_initial', type=float, default=25)
    parser.add_argument('--div_factor_final', type=float, default=1000)
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--num_workers', type=int, default=0)

    return parser.parse_args()


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, scheduler, lambda_subj, device):
    model.train()
    total_loss, cls_loss_sum, subj_loss_sum = 0., 0., 0.
    cls_correct, subj_correct, total = 0, 0, 0

    for data, target, subject_ids, bt_mask in loader:
        data = data.to(device, dtype=torch.float32)
        bt_mask = bt_mask.to(device)
        target = target.to(device)
        subject_ids = subject_ids.to(device)

        B, T_pad, X, Y, Z = data.shape

        # Flatten batch and time, keep only valid timepoints
        data_flat = data.view(B * T_pad, X, Y, Z)
        mask_flat = bt_mask.view(B * T_pad)
        valid_idx = mask_flat.nonzero(as_tuple=True)[0]

        if valid_idx.numel() == 0:
            continue

        data_valid = data_flat[valid_idx].unsqueeze(-1)  # (N_valid, X, Y, Z, 1)

        outputs = model(data_valid)
        logits_cls = outputs["logits_cls"]
        logits_subj = outputs["logits_subj"]

        # Aggregate per-timepoint logits back to per-sample (temporal mean)
        b_idx = valid_idx // T_pad
        num_classes = logits_cls.shape[-1]
        num_subjects = logits_subj.shape[-1]

        logits_cls_agg = torch.zeros(B, num_classes, device=device)
        logits_subj_agg = torch.zeros(B, num_subjects, device=device)
        counts = torch.zeros(B, device=device)

        logits_cls_agg.index_add_(0, b_idx, logits_cls)
        logits_subj_agg.index_add_(0, b_idx, logits_subj)
        counts.index_add_(0, b_idx, torch.ones_like(b_idx, dtype=torch.float32))
        counts = counts.clamp(min=1.0)

        logits_cls_agg = logits_cls_agg / counts.unsqueeze(-1)
        logits_subj_agg = logits_subj_agg / counts.unsqueeze(-1)

        cls_loss = F.cross_entropy(logits_cls_agg, target)
        subj_loss = F.cross_entropy(logits_subj_agg, subject_ids)
        loss = cls_loss + lambda_subj * subj_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler.last_epoch < scheduler.total_steps - 1:
            scheduler.step()

        pred_cls = logits_cls_agg.argmax(dim=1)
        pred_subj = logits_subj_agg.argmax(dim=1)

        total_loss += loss.item()
        cls_loss_sum += cls_loss.item()
        subj_loss_sum += subj_loss.item()
        cls_correct += pred_cls.eq(target).sum().item()
        subj_correct += pred_subj.eq(subject_ids).sum().item()
        total += B

    n = len(loader)
    return (total_loss / n, cls_loss_sum / n, subj_loss_sum / n,
            cls_correct / max(total, 1), subj_correct / max(total, 1))


@torch.no_grad()
def validate(model, loader, lambda_subj, device):
    model.eval()
    total_loss, cls_loss_sum, subj_loss_sum = 0., 0., 0.
    cls_correct, subj_correct, total = 0, 0, 0

    for data, target, subject_ids, bt_mask in loader:
        data = data.to(device, dtype=torch.float32)
        bt_mask = bt_mask.to(device)
        target = target.to(device)
        subject_ids = subject_ids.to(device)

        B, T_pad, X, Y, Z = data.shape

        data_flat = data.view(B * T_pad, X, Y, Z)
        mask_flat = bt_mask.view(B * T_pad)
        valid_idx = mask_flat.nonzero(as_tuple=True)[0]

        if valid_idx.numel() == 0:
            continue

        data_valid = data_flat[valid_idx].unsqueeze(-1)

        outputs = model(data_valid)
        logits_cls = outputs["logits_cls"]
        logits_subj = outputs["logits_subj"]

        b_idx = valid_idx // T_pad
        num_classes = logits_cls.shape[-1]
        num_subjects = logits_subj.shape[-1]

        logits_cls_agg = torch.zeros(B, num_classes, device=device)
        logits_subj_agg = torch.zeros(B, num_subjects, device=device)
        counts = torch.zeros(B, device=device)

        logits_cls_agg.index_add_(0, b_idx, logits_cls)
        logits_subj_agg.index_add_(0, b_idx, logits_subj)
        counts.index_add_(0, b_idx, torch.ones_like(b_idx, dtype=torch.float32))
        counts = counts.clamp(min=1.0)

        logits_cls_agg = logits_cls_agg / counts.unsqueeze(-1)
        logits_subj_agg = logits_subj_agg / counts.unsqueeze(-1)

        cls_loss = F.cross_entropy(logits_cls_agg, target)
        subj_loss = F.cross_entropy(logits_subj_agg, subject_ids)
        loss = cls_loss + lambda_subj * subj_loss

        pred_cls = logits_cls_agg.argmax(dim=1)
        pred_subj = logits_subj_agg.argmax(dim=1)

        total_loss += loss.item()
        cls_loss_sum += cls_loss.item()
        subj_loss_sum += subj_loss.item()
        cls_correct += pred_cls.eq(target).sum().item()
        subj_correct += pred_subj.eq(subject_ids).sum().item()
        total += B

    n = len(loader)
    return (total_loss / n, cls_loss_sum / n, subj_loss_sum / n,
            cls_correct / max(total, 1), subj_correct / max(total, 1))


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    set_seed(args.seed, device)

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'results.csv')

    # Dataset
    dataset = HCPDataset(data_dir=args.data_dir)
    print(f"Loaded {len(dataset)} samples")

    # Cross-validation
    y = np.array(dataset.event_labels)
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    all_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{args.num_folds}")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
        print(f"{'='*60}")

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=fmri_collate_fn,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            collate_fn=fmri_collate_fn,
        )

        num_subjects = len(dataset.subject_map)
        num_events = len(dataset.event_map)

        model = SubjectAwareSparceiver(
            input_channels=1,
            input_axis=3,
            num_freq_bands=args.num_freq_bands,
            max_freq=args.max_freq,
            depth=args.depth,
            num_latents=args.num_latents,
            latent_dim=args.latent_dim,
            cross_heads=args.cross_heads,
            latent_heads=args.latent_heads,
            cross_dim_head=args.latent_dim,
            latent_dim_head=args.latent_dim,
            num_classes=num_events,
            num_subjects=num_subjects,
            weight_tie_layers=args.weight_tie_layers,
            self_per_cross_attn=args.self_per_cross_attn,
            subj_extraction_layer=args.subj_extraction_layer,
            subj_extraction_block=args.subj_extraction_block,
            topk=args.topk,
            maxprop=args.maxprop,
        ).to(device)

        optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.num_epochs,
            pct_start=args.warmup_pct,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=args.div_factor_initial,
            final_div_factor=args.div_factor_final,
        )

        best_val_acc = 0.
        start_time = time.time()

        for epoch in range(args.num_epochs):
            tr_loss, tr_cls, tr_subj, tr_acc, tr_sacc = train_one_epoch(
                model, train_loader, optimizer, scheduler, args.lambda_subj, device,
            )
            vl_loss, vl_cls, vl_subj, vl_acc, vl_sacc = validate(
                model, val_loader, args.lambda_subj, device,
            )

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                save_checkpoint(args.output_dir, {
                    'epoch': epoch + 1,
                    'fold': fold,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_acc': vl_acc,
                }, f'best_fold{fold}.pth')

            if epoch % 5 == 0 or epoch == args.num_epochs - 1:
                print(f"  Epoch {epoch+1:3d}/{args.num_epochs} | "
                      f"Train acc={tr_acc:.4f} subj={tr_sacc:.4f} loss={tr_loss:.4f} | "
                      f"Val acc={vl_acc:.4f} subj={vl_sacc:.4f} loss={vl_loss:.4f}")

        elapsed = time.time() - start_time
        print(f"  Fold {fold+1} done in {elapsed/60:.1f} min | Best val acc: {best_val_acc:.4f}")

        all_results.append({
            'fold': fold,
            'best_val_acc': best_val_acc,
            'final_val_acc': vl_acc,
            'final_train_acc': tr_acc,
        })

    # Save summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    print(f"Mean best val acc: {results_df['best_val_acc'].mean():.4f} "
          f"(+/- {results_df['best_val_acc'].std():.4f})")


if __name__ == '__main__':
    main()
