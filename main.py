"""Single entry point for the experiments of paper Sections 4 and 5.

    python main.py --dataname adult --method blackbox --mode train      # the black box
    python main.py --dataname adult --method vae      --mode train      # TABCF's VAE
    python main.py --dataname adult --method flowcf   --mode sample  --num_samples 1000
    python main.py --dataname adult --method flowcf   --mode evaluate --num_samples 1000

`run_benchmark.py` calls this for every method in turn.
"""

import argparse

import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(description='FlowCF experiments')

    parser.add_argument('--dataname', type=str, default='adult', help='Dataset name.')
    parser.add_argument('--method', type=str, default='flowcf',
                        help='flowcf | tabcf | dicoflex | wachter | dice | revise | cchvae '
                             '| blackbox | vae')
    parser.add_argument('--mode', type=str, default='sample', help='train | sample | evaluate.')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Factual instances to explain (-1 = all).')
    parser.add_argument('--save_path', type=str, default='counterfactual_results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--verbose', type=str2bool, default=False)
    parser.add_argument('--immutable', type=str2bool, default=False,
                        help='Enforce the dataset\'s immutable features (off in the paper\'s '
                             'experiments, so all methods face the same problem).')
    parser.add_argument('--total_CFs', type=int, default=1,
                        help='Counterfactuals per instance (1 in the paper).')

    # TABCF's VAE. These name its checkpoint directory, so sampling must be given the
    # same values it was trained with; the defaults are the ones used in the paper.
    parser.add_argument('--max_beta', type=float, default=1e-2)
    parser.add_argument('--min_beta', type=float, default=1e-5)
    parser.add_argument('--lambd', type=float, default=0.7)
    parser.add_argument('--kl_weight', type=float, default=-1)
    parser.add_argument('--gumbel_softmax', type=str2bool, default=True)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--sigmoid', type=str2bool, default=True)
    parser.add_argument('--reparam', type=str2bool, default=True)
    parser.add_argument('--num_reconstr_loss', type=str, default='L2')
    parser.add_argument('--num_encoding', type=str, default='min_max_torch')

    # Gradient-based baselines (DiCE, TABCF): their published objective.
    parser.add_argument('--dice_method', type=str, default='gradient')
    parser.add_argument('--dice_optimization', type=str2bool, default=True)
    parser.add_argument('--dice_post_hoc_sparsity', type=str2bool, default=False)
    parser.add_argument('--decode_before_loss', type=str2bool, default=True)
    parser.add_argument('--proximity_weight_input', type=float, default=1.0)
    parser.add_argument('--proximity_weight_latent', type=float, default=1.0)
    parser.add_argument('--proximity_latent_loss', type=str, default='L2')
    parser.add_argument('--proximity_input_loss', type=str, default=None)
    parser.add_argument('--plausibility_weight_latent', type=float, default=-1.0)
    parser.add_argument('--plausibility_latent_loss', type=str, default='GMM')
    parser.add_argument('--min_iter', type=int, default=500)
    parser.add_argument('--max_iter', type=int, default=5000)
    parser.add_argument('--plot_gradients', type=str2bool, default=False)

    return parser.parse_args()


def execute_function(method, mode):
    if mode == 'evaluate':
        from evaluate import main as fn
        return fn
    if method == 'blackbox':
        from black_box import main as fn
        return fn
    if method == 'vae':
        from baselines.tabcf.vae.main import main as fn
        return fn

    import importlib
    return importlib.import_module(f'baselines.{method}.sample').main


if __name__ == '__main__':
    args = get_args()
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    execute_function(args.method, args.mode)(args)
