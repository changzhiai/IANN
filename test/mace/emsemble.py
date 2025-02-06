from ase.io import read, write, Trajectory
import numpy as np
import torch
import toml
import argparse
from pathlib import Path
import logging
from iann.models.calculators import MLCalculator,EnsembleCalculator
from iann.models.mace import MACE
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.db import connect
import numpy as np
import pandas as pd
import os
import subprocess

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

class EnergyObservor:
    def __init__(self, atoms):
        self.atoms = atoms
        print("Energy observor")

    def __call__(self, threshold=0):
        energy = self.atoms.get_potential_energy()
        forces = self.atoms.get_forces()
        print(f'energy: {energy}')
        if energy > threshold or energy<-10000:
            raise ValueError('energy is too large or too low')

def plot_fitting(dft_energies, nnp_energies, fig_name='prediction.png', output_dir=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    rmse = mean_squared_error(dft_energies, nnp_energies, squared=False)
    m, b = np.polyfit(dft_energies, nnp_energies, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.array(dft_energies), m * np.array(dft_energies) + b)
    plt.plot(dft_energies, nnp_energies, 'ob',mfc='none')
    plt.xlabel("$E_b$(DFT) (eV)")
    plt.ylabel("$E_b$(NNP) (eV)")
    plt.title("fit using {} data points.".format(len(nnp_energies)))
    X = dft_energies
    Y = nnp_energies
    if np.size(X) and np.size(Y) != 0:
        e_range = max(np.append(X, Y)) - min(np.append(X, Y))
        rmin = min(np.append(X, Y)) - 0.05 * e_range
        rmax = max(np.append(X, Y)) + 0.05 * e_range
    else:
        rmin = -10
        rmax = 10
    linear_fit = np.arange(rmin - 10, rmax + 10, 1)
    ax.plot(linear_fit, linear_fit, 'r')
    ax.axis([rmin, rmax, rmin, rmax])
    ax.text(0.95,
            0.01,
            "RMSE = {:.3f} meV".format(rmse*1000),
            verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes,
            fontsize=12)
    ticks = False
    if ticks:
        plt.xticks(np.arange(min(X), max(Y), 0.2))
        plt.yticks(np.arange(min(X), max(Y), 0.2))
    plt.show()
    fig.savefig(f'{path}/{output_dir}/{fig_name}')

def setup_seed(seed):
     torch.manual_seed(seed)
     if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="graph neural networks", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=f"{path}/arguments_predict.toml",
        help="Path to config file. e.g. 'arguments.toml'"
    )

    return parser.parse_args(arg_list)

def update_namespace(ns, d):
    for k, v in d.items():
        ns.__dict__[k] = v

class CallsCounter:
    def __init__(self, func):
        self.calls = 0
        self.func = func
    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.func(*args, **kwargs)

def generate_run_config(params):
    # with open(os.path.join(params['run_dir'], "run.toml"), 'w') as f:
    with open(os.path.join(path, "run.toml"), 'w') as f:
        toml.dump(params, f)
    return params

def read_df(args):
    df = pd.read_csv(f'{path}/{args.save_csv}')
    # df = df.loc[df['converged']==True]
    dft_energies = df['dft_energies']
    nnp_energies = df['nnp_energies']
    fig_name = args.fig_name
    return dft_energies, nnp_energies, fig_name

def get_init_bad_structs(args):
    df = pd.read_csv(args.save_csv)
    df = df[(df['dft_energies']-df['nnp_energies']>0.02) | (df['converged']==False)]
    images = read(args.all_init_dft_dir, ':')
    lists = list(df.index.values)
    images_bad = [images[i] for i in lists]
    print(len(images_bad))
    print([images[i].get_chemical_formula() for i in lists])
    write('images_bad.traj', images_bad)
    return df

def main(read_csv=False):
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
    update_namespace(args, params)
    if read_csv:
        print('Reading csv')
        # get_init_bad_structs(args)
        return read_df(args)

    _ = generate_run_config(params)
    setup_seed(args.random_seed)
    # set logger
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)
    runHandler = logging.FileHandler(f'{path}/{args.output_dir}/args.predict_log', mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    errorHandler = logging.FileHandler(f'{path}/{args.output_dir}/args.error_log', mode='w')
    errorHandler.setLevel(logging.WARNING)
    errorHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    logger.addHandler(runHandler)
    logger.addHandler(errorHandler)
    logger.addHandler(logging.StreamHandler())
    logger.warning = CallsCounter(logger.warning)
    logger.info = CallsCounter(logger.info)
    # load models
    models = []
    for each in args.load_models:
        state_dict = torch.load(each, map_location=torch.device(args.device))
        model = MACE(
            cutoff = state_dict['cutoff'],
            num_interactions = state_dict['num_layer'],
            num_features = state_dict['node_size'],
            correlation = 3,
            species = None,
            compute_forces = bool(args.forces_weight),
        )
        model.to(args.device)
        model.load_state_dict(state_dict["model"])
        models.append(model)

    calc = EnsembleCalculator(models)

    # state_dict = torch.load(f'{path}/{args.load_model}', map_location=torch.device(args.device)) 
    # model = MACE(
    #     cutoff = args.cutoff,
    #     num_interactions = args.num_interactions,
    #     num_features = args.node_size,
    #     correlation = 3,
    #     species = None,
    #     compute_forces = bool(args.forces_weight),
    # )
    # model.to(args.device)
    # model.load_state_dict(state_dict["model"])
    # calc = MLCalculator(model)

    images = read(f'{path}/{args.dataset}', ':')
    count = 0
    formulas = []
    dft_energies = []
    nnp_energies = []
    E_per_atoms = args.E_per_atoms
    for atoms in images:
        E_dft = atoms.get_potential_energy()
        if E_per_atoms:
            E_dft = E_dft/len(atoms)
        dft_energies.append(E_dft)
    for atoms in images:
        formula = atoms.get_chemical_formula(mode='hill')
        formulas.append(formula)
        atoms.calc = calc
        logger.info('finishing: {}'.format(count))
        E_nnp = atoms.get_potential_energy()
        if E_per_atoms:
            E_nnp = E_nnp/len(atoms)
        nnp_energies.append(E_nnp)
        count += 1
    tuples = {
            'formula': formulas,
            'dft_energies': dft_energies,
            'nnp_energies':nnp_energies,
            }
    df = pd.DataFrame(tuples)
    df.to_csv(f'{path}/{args.output_dir}/{args.save_csv}')
    return df['dft_energies'], df['nnp_energies'], args.fig_name, args.output_dir

if __name__ == "__main__":
    dft_energies, nnp_energies, fig_name, output_dir = main(read_csv=False)
    plot_fitting(dft_energies, nnp_energies, fig_name=fig_name, output_dir=output_dir)
    print('Predict done!')
