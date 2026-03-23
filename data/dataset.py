"""HCP1200 task-fMRI dataset for trial-level voxel decoding."""

import glob
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset


# HCP task-condition mapping (7 tasks, 24 conditions)
EVENT_DICT = {
    'tfMRI_GAMBLING_LR':    ['win', 'loss'],
    'tfMRI_SOCIAL_LR':      ['rnd', 'mental'],
    'tfMRI_MOTOR_LR':       ['rh', 't', 'lh', 'lf', 'rf'],
    'tfMRI_EMOTION_LR':     ['fear', 'neut'],
    'tfMRI_LANGUAGE_LR':    ['math', 'story'],
    'tfMRI_WM_LR':          ['2bk_places', '0bk_body', '2bk_faces', '0bk_faces',
                              '2bk_tools', '0bk_places', '2bk_body', '0bk_tools'],
    'tfMRI_RELATIONAL_LR':  ['match', 'relation'],
}


class HCPDataset(Dataset):
    """HCP1200 trial-level fMRI dataset.

    Loads pre-extracted trial volumes (.npy files) organized as:
        data_dir/{subject_id}/{task_name}/{condition}/{subject}_{condition}_{idx}.npy

    Each .npy file is a 4-D array (X, Y, Z, T) representing one trial.

    Args:
        data_dir: Root directory containing subject folders.
        task_events: Dict mapping task names to condition lists.
            Defaults to EVENT_DICT (all 7 HCP tasks).
    """

    def __init__(self, data_dir, task_events=None):
        self.data_dir = Path(data_dir)

        if task_events is None:
            task_events = EVENT_DICT

        self.file_list = []
        self.event_labels = []
        self.subject_labels = []

        # Build condition -> label mapping
        self.event_map = {}
        label_idx = 0
        for task_name in sorted(task_events.keys()):
            for event in sorted(task_events[task_name]):
                task_cond = f"{task_name}_{event}"
                self.event_map[task_cond] = label_idx
                label_idx += 1

        # Collect files and validate
        all_subjects_set = set()
        total_files = 0
        valid_files = 0

        for task_name, events in task_events.items():
            for event in events:
                pattern = str(self.data_dir / "*" / task_name / event / f"*_{event}_*.npy")
                files = glob.glob(pattern)

                for filepath in files:
                    total_files += 1
                    try:
                        test_data = np.load(filepath)
                        if test_data.shape[-1] == 0 or test_data.size == 0:
                            continue
                        if len(test_data.shape) != 4:
                            continue
                    except Exception:
                        continue

                    subject_id = Path(filepath).parts[-4]
                    all_subjects_set.add(subject_id)
                    self.file_list.append(filepath)

                    task_cond = f"{task_name}_{event}"
                    self.event_labels.append(self.event_map[task_cond])
                    valid_files += 1

        # Build subject -> index mapping
        self.subject_map = {subj: idx for idx, subj in enumerate(sorted(all_subjects_set))}

        for filepath in self.file_list:
            subject_id = Path(filepath).parts[-4]
            self.subject_labels.append(self.subject_map[subject_id])

        print(f"Dataset: {valid_files}/{total_files} valid samples, "
              f"{len(self.event_map)} conditions, {len(self.subject_map)} subjects")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        arr = np.load(filepath).astype(np.float32)  # (X, Y, Z, T)
        data = torch.from_numpy(arr).permute(3, 0, 1, 2)  # -> (T, X, Y, Z)
        event_label = self.event_labels[idx]
        subject_label = self.subject_labels[idx]
        return data, event_label, subject_label


def fmri_collate_fn(batch):
    """Custom collate that zero-pads variable-length temporal dimension.

    Returns:
        batch_vols: (B, T_max, X, Y, Z) zero-padded volumes.
        labels: (B,) condition labels.
        subjects: (B,) subject labels.
        bt_mask: (B, T_max) boolean mask for valid timepoints.
    """
    T_list = [v.shape[0] for (v, _, _) in batch]
    T_max = max(T_list)
    X, Y, Z = batch[0][0].shape[1:]
    B = len(batch)

    batch_vols = torch.zeros(B, T_max, X, Y, Z, dtype=torch.float32)
    bt_mask = torch.zeros(B, T_max, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    subjects = torch.zeros(B, dtype=torch.long)

    for i, (vols, label, subj) in enumerate(batch):
        T_i = vols.shape[0]
        batch_vols[i, :T_i] = vols
        bt_mask[i, :T_i] = True
        labels[i] = label
        subjects[i] = subj

    return batch_vols, labels, subjects, bt_mask
