"""Shared data plumbing for the benchmark harness.

`get_input_generate` returns the dataset `info`, the numerical/categorical inverse
transforms, and the labels `y` for a method to build counterfactuals. This is the
VAE-FREE version used by every non-tabcf baseline (dice, wachter, revise, cchvae).

The tabcf method needs the trained VAE in addition; it has its own
`get_input_generate` in baselines/tabcf/latent_utils.py that wraps this one.
The return shape (10-tuple) is kept identical so baselines can unpack it the same
way; the two VAE slots (train_z, test_z) are None here.
"""

import os
import json

from utils_train import preprocess, get_name_form_args


def get_input_generate(args):
    dataname = args.dataname
    num_encoding = args.num_encoding

    output_file_name_vae = get_name_form_args(args)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f'data/{dataname}'

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']

    ckpt_dir = f'{curr_dir}/ckpt/{dataname}/'
    vae_ckpt_dir = f'{curr_dir}/vae/ckpt/{dataname}/{output_file_name_vae}'

    _, _, _, _, num_inverse, cat_inverse, y = preprocess(
        dataset_dir, task_type=task_type, inverse=True, num_encoding=num_encoding)

    return (None, None, curr_dir, dataset_dir, ckpt_dir, vae_ckpt_dir,
            info, num_inverse, cat_inverse, y)
