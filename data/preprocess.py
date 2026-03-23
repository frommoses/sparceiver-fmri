"""Preprocessing utilities for HCP1200 fMRI data.

Extracts trial-level volumes from z-scored fMRI data using event onset files,
and optionally converts the .npy trial files into a single HDF5 archive.
"""

import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from tqdm import tqdm

TR = 0.72  # HCP repetition time in seconds

CONDITION_DICT = {
    'tfMRI_GAMBLING_LR':    ['loss_event', 'win', 'loss', 'win_event', 'neut_event'],
    'tfMRI_SOCIAL_LR':      ['rnd', 'mental_resp', 'mental', 'other_resp'],
    'tfMRI_MOTOR_LR':       ['rh', 't', 'lh', 'lf', 'rf', 'cue'],
    'tfMRI_EMOTION_LR':     ['fear', 'neut'],
    'tfMRI_LANGUAGE_LR':    ['math', 'present_story', 'question_math', 'response_math',
                              'story', 'question_story', 'cue', 'response_story', 'present_math'],
    'tfMRI_WM_LR':          ['0bk_err', '2bk_places', '0bk_body', '2bk_cor', '0bk_cor',
                              '2bk_faces', '0bk_faces', '2bk_err', 'all_bk_err', 'all_bk_cor',
                              '2bk_tools', '2bk_nlr', '0bk_places', '0bk_nlr', '2bk_body', '0bk_tools'],
    'tfMRI_RELATIONAL_LR':  ['match', 'relation', 'error'],
}


def extract_trial_volumes(subject_id, task_name, condition_name, data_dir, save_dir, save=True):
    """Extract trial-level fMRI volumes for a single subject/task/condition.

    Reads the z-scored fMRI NIfTI and event onset file, then slices volumes
    corresponding to each trial.

    Args:
        subject_id: HCP subject ID string.
        task_name: HCP task name (e.g., 'tfMRI_EMOTION_LR').
        condition_name: Condition within the task (e.g., 'fear').
        data_dir: Root directory of preprocessed HCP results.
        save_dir: Output directory for .npy trial files.
        save: Whether to save files to disk.

    Returns:
        None if successful, dict with error info if failed.
    """
    output_dir = Path(save_dir, subject_id, task_name, condition_name)
    csv_path = output_dir / f"{subject_id}_{task_name}.csv"

    if csv_path.exists():
        return None

    task_dir = Path(data_dir, subject_id, task_name)
    if not task_dir.exists():
        return {'subject': subject_id, 'task': task_name, 'condition': condition_name, 'error': 'task_dir'}

    ev_dir = Path(task_dir, 'EVs')
    if not ev_dir.exists():
        return {'subject': subject_id, 'task': task_name, 'condition': condition_name, 'error': 'ev_dir'}

    fmri_file = Path(task_dir, 'epi_final_zscore.nii.gz')
    if not fmri_file.exists():
        return {'subject': subject_id, 'task': task_name, 'condition': condition_name, 'error': 'fmri_file'}

    onset_file = ev_dir / f'{condition_name}.txt'
    if not onset_file.exists():
        return {'subject': subject_id, 'task': task_name, 'condition': condition_name, 'error': 'onset_file'}

    onset_df = pd.read_csv(onset_file, sep=r'\s+', header=None, names=['onset', 'duration', 'weight'])

    try:
        img = nib.load(fmri_file)
        data = img.get_fdata()
    except EOFError:
        return {'subject': subject_id, 'task': task_name, 'condition': condition_name, 'error': 'corrupted_file'}
    except Exception as e:
        return {'subject': subject_id, 'task': task_name, 'condition': condition_name, 'error': str(e)}

    metadata_list = []
    num = 0

    if save:
        transpose_dir = Path(output_dir, "transpose")
        output_dir.mkdir(parents=True, exist_ok=True)
        transpose_dir.mkdir(parents=True, exist_ok=True)

    for trial_idx, (_, row) in enumerate(onset_df.iterrows(), start=1):
        onset = row['onset']
        duration = row['duration']

        start_idx = int(onset / TR)
        end_idx = int(np.ceil((onset + duration) / TR))

        volumes = data[..., start_idx:end_idx]
        volumes_t = np.transpose(volumes, (3, 0, 1, 2))

        if save:
            output_file = output_dir / f"{subject_id}_{condition_name}_{num}.npy"
            np.save(output_file, volumes)

            output_file_t = transpose_dir / f"{subject_id}_{condition_name}_t_{num}.npy"
            np.save(output_file_t, volumes_t)

            num += 1

            metadata_list.append({
                'subject_id': subject_id,
                'task_name': task_name,
                'condition': condition_name,
                'trial_idx': trial_idx,
                'onset_raw': onset,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'n_volumes': end_idx - start_idx,
                'shape': str(volumes.shape),
                'file_path': str(output_file.relative_to(output_dir)),
            })

    if save and metadata_list:
        df = pd.DataFrame(metadata_list)
        df.to_csv(csv_path, index=False)

    return None


def preprocess_all(data_dir, save_dir, condition_dict=None):
    """Extract trial volumes for all subjects and conditions.

    Args:
        data_dir: Root directory of preprocessed HCP results.
        save_dir: Output directory for .npy trial files.
        condition_dict: Task-condition mapping. Defaults to CONDITION_DICT.
    """
    if condition_dict is None:
        condition_dict = CONDITION_DICT

    error_list = []

    for subject in tqdm(sorted(os.listdir(data_dir)), desc="Subjects"):
        for task, conditions in condition_dict.items():
            for condition in conditions:
                result = extract_trial_volumes(
                    subject_id=subject,
                    task_name=task,
                    condition_name=condition,
                    data_dir=data_dir,
                    save_dir=save_dir,
                    save=True,
                )
                if result is not None:
                    error_list.append(result)

    if error_list:
        error_df = pd.DataFrame(error_list)
        error_path = os.path.join(save_dir, 'error_log.csv')
        error_df.to_csv(error_path, index=False)
        print(f"Completed with {len(error_list)} errors. Log: {error_path}")
    else:
        print("All subjects processed successfully.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess HCP fMRI data')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory of preprocessed HCP results')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Output directory for trial .npy files')
    args = parser.parse_args()

    preprocess_all(args.data_dir, args.save_dir)
