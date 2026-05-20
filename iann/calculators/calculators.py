from ase.calculators.calculator import Calculator, all_changes
from iann.data import AseDataReader
import numpy as np
import torch

def _load_model(model_path, device, compute_forces, **kwargs):
    """Load model from path and determine its type."""
    state_dict = torch.load(model_path, map_location=device)
    
    # Determine model type from state dict
    if "model_type" in state_dict:
        model_type = state_dict["model_type"].lower()
    else:
        # Try to determine from model architecture
        if "num_layers" in state_dict:
            model_type = "painn"
        elif "irreps" in state_dict:
            model_type = "nequip"
        elif "correlation" in state_dict:
            model_type = "mace"
        elif "transformer" in state_dict:
            model_type = "equiformerv2"
        else:
            raise ValueError("Could not determine model type from state dict!")

    # Clean up kwargs
    model_kwargs = kwargs.copy()
    
    # compute_forces can be toggled at inference time if the model supports it
    forces_enabled = model_kwargs.pop("compute_forces", state_dict.get("compute_forces", False))
    if compute_forces is not None:
        forces_enabled = compute_forces

    # Create appropriate model
    if model_type == "painn":
        from iann.models.painn import PaiNN
        model = PaiNN(
            num_layers=state_dict["num_layers"],
            num_channels=state_dict["num_channels"],
            cutoff=state_dict["cutoff"],
            compute_forces=forces_enabled,
            **model_kwargs,
        )
    elif model_type == "nequip":
        from iann.models.nequip import NequIP
        # Structural parameters are fixed at training time and should not be overridden
        model_kwargs.pop("lmax", None)
        model_kwargs.pop("parity", None)
        model_kwargs.pop("use_cue", None)
        
        lmax_sd = state_dict.get("lmax")
        if lmax_sd is None: lmax_sd = 2
        parity_sd = state_dict.get("parity")
        if parity_sd is None: parity_sd = True
        use_cue_sd = state_dict.get("use_cue", False)

        model = NequIP(
            num_layers=state_dict["num_layers"],
            num_channels=state_dict["num_channels"],
            cutoff=state_dict["cutoff"],
            lmax=lmax_sd,
            parity=parity_sd,
            use_cue=use_cue_sd,
            compute_forces=forces_enabled,
            **model_kwargs,
        )
    elif model_type == "mace":
        from iann.models.mace import MACE
        # Structural parameters are fixed at training time and should not be overridden
        model_kwargs.pop("lmax", None)
        model_kwargs.pop("use_cue", None)
        
        lmax_sd = state_dict.get("lmax")
        if lmax_sd is None: lmax_sd = 3
        use_cue_sd = state_dict.get("use_cue", False)
        
        model = MACE(
            num_layers=state_dict["num_layers"],
            num_channels=state_dict["num_channels"],
            cutoff=state_dict["cutoff"],
            lmax=lmax_sd,
            use_cue=use_cue_sd,
            compute_forces=forces_enabled,
            **model_kwargs,
        )
    elif model_type == "equiformerv2":
        from iann.models.equiformerV2 import EquiformerV2
        # Structural parameters are fixed at training time and should not be overridden
        model_kwargs.pop("lmax", None)
        model_kwargs.pop("parity", None)
        
        lmax_sd = state_dict.get("lmax")
        if lmax_sd is None: lmax_sd = 2
        parity_sd = state_dict.get("parity")
        if parity_sd is None: parity_sd = True
        
        model = EquiformerV2(
            num_layers=state_dict["num_layers"],
            num_channels=state_dict["num_channels"],
            cutoff=state_dict["cutoff"],
            # EquiformerV2 does not support use_cue
            compute_forces=forces_enabled,
            **model_kwargs,
        )
    elif model_type == "equiformerv3":
        from iann.models.equiformerV3 import EquiformerV3
        # Structural parameters are fixed at training time and should not be overridden
        model_kwargs.pop("parity", None)
        model_kwargs.pop("use_cue", None)

        model = EquiformerV3(
            num_layers=state_dict["num_layers"],
            num_channels=state_dict["num_channels"],
            cutoff=state_dict["cutoff"],
            compute_forces=forces_enabled,
            **model_kwargs,
        )
    elif model_type == "fastpot":
        from iann.models.fastpot import FastPot
        model = FastPot(
            num_layers=state_dict["num_layers"],
            num_channels=state_dict["num_channels"],
            cutoff=state_dict["cutoff"],
            compute_forces=forces_enabled,
            **model_kwargs,
        )
    elif model_type == "demo":
        from iann.models.demo import Demo
        model = Demo(
            num_layers=state_dict["num_layers"],
            num_channels=state_dict["num_channels"],
            cutoff=state_dict["cutoff"],
            compute_forces=forces_enabled,
            **model_kwargs,
        )
    elif model_type == "uma":
        from iann.models.uma import UMA
        model_kwargs.pop("parity", None)
        model_kwargs.pop("use_cue", None)
        model = UMA(
            num_layers=state_dict["num_layers"],
            num_channels=state_dict["num_channels"],
            cutoff=state_dict["cutoff"],
            device=device,
            compute_forces=forces_enabled,
            **model_kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Please choose from: painn, nequip, mace, equiformerV2, equiformerV3, uma, fastpot, and demo!")
    
    model.load_state_dict(state_dict["model"])
    model.to(device)
    return model

class MLCalculator(Calculator):
    """
    Machine learning calculator for a single model.
    """
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model_path=None,
        model=None,
        config=None,
        energy_scale=1.0,
        forces_scale=1.0,
        device=None,
        verbose=False,
        **kwargs):
        """
        Parameters
        ----------
        model_path : str
            Path to the model (only one model).
        model : iann.models.Model
            Model to use.
        config : dict
            Configuration for the calculator.
        energy_scale : float
            Energy scale.
        forces_scale : float
            Forces scale.
        device : str
            Device to use.
        verbose : bool
            Verbosity.
        
        Returns
        -------
        Dict[str, array]
            Dictionary with keys 'energy', 'forces'.
        """
        super().__init__(**kwargs)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        if verbose:
            print(f"Using device: {self.device}")

        if config is not None:
            self.config = config
        else:
            self.config = {}

        if "compute_forces" in self.config:
            self.compute_forces = self.config["compute_forces"]
        else:
            if "compute_forces" in kwargs:
                self.compute_forces = kwargs["compute_forces"]
                kwargs.pop("compute_forces")
            else:
                self.compute_forces = None

        if "compute_stress" in self.config:
            self.compute_stress = self.config["compute_stress"]
        else:
            if "compute_stress" in kwargs:
                self.compute_stress = kwargs["compute_stress"]
            else:
                self.compute_stress = None

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = _load_model(model_path, self.device, self.compute_forces, **kwargs)
            self.compute_forces = self.model.compute_forces
        else:
            raise ValueError("Either model or model_path must be provided")

        self.cutoff = self.model.cutoff
        self.ase_data_reader = AseDataReader(self.cutoff, self.compute_forces)
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale

        if verbose:
            print(f"Model: {self.model}")
            print(f"Cutoff: {self.cutoff}")
            print(f"Compute forces: {self.compute_forces}")
            print(f"Compute stress: {self.compute_stress}")
            print(f"Energy scale: {self.energy_scale}")
            print(f"Forces scale: {self.forces_scale}")


    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        """
        Parameters
        ----------
        atoms : ase.Atoms
            ASE atoms object.
        properties : list of str
            energy, forces, and stress are supported.
        system_changes : list of str
            List of changes for ASE.
        """
        if atoms is not None:
            self.atoms = atoms.copy()       

        model_inputs = self.ase_data_reader(self.atoms)
        
        if self.device == 'cuda':
            model_inputs = model_inputs.to(self.device)

        # Ensure model is in evaluation mode for deterministic inference
        self.model.eval()
        
        model_results = self.model(model_inputs)

        results = {}
        results["energy"] = model_results.energy[0].detach().cpu().numpy().item()
        if self.compute_forces:
            results["forces"] = model_results.forces.detach().cpu().numpy() * self.forces_scale
        if self.compute_stress:
            results["stress"] = model_results.stress.detach().cpu().numpy()
    
        self.results = results


class EnsembleCalculator(Calculator):
    """
    Ensemble calculator for multiple models.
    """
    implemented_properties = ["energy", "forces", "ensemble"]

    def __init__(
        self,
        model_paths=None,
        models=None,
        config=None,
        energy_scale=1.0,
        forces_scale=1.0,
        device=None,
        verbose=False,
        **kwargs):
        """
        Parameters
        ----------
        model_paths : list of str
            Paths to the models (several models for ensemble).
        models : list of iann.models.Model
            Models to use.
        config : dict
            Configuration for the calculator.
        energy_scale : float
            Energy scale.
        forces_scale : float
            Forces scale.
        device : str
            Device to use.
        verbose : bool
            Verbosity.

        Returns
        -------
        Dict[str, array]
            Dictionary with keys 'energy', 'forces', 'ensemble'.
        """
        super().__init__(**kwargs)

        if config is not None:
            self.config = config
        else:
            self.config = {}

        if "compute_forces" in self.config:
            self.compute_forces = self.config["compute_forces"]
        else:
            if "compute_forces" in kwargs:
                self.compute_forces = kwargs["compute_forces"]
                kwargs.pop("compute_forces")
            else:
                self.compute_forces = None

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if model_paths is not None:
            self.models = [_load_model(model_path, self.device, self.compute_forces, **kwargs) for model_path in model_paths]
        elif models is not None:
            self.models = models
        else:
            raise ValueError("Either model_paths or models must be provided")

        self.cutoff = self.models[0].cutoff
        self.compute_forces = self.models[0].compute_forces
        self.ase_data_reader = AseDataReader(self.cutoff, self.compute_forces)
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale

        if verbose:
            print(f"Ensemble calculator initialized with {len(self.models)} models")
            print(f"Cutoff: {self.cutoff}")
            print(f"Compute forces: {self.compute_forces}")
            print(f"Energy scale: {self.energy_scale}")
            print(f"Forces scale: {self.forces_scale}")
    
    def get_ensemble(self):
        """Get the calculated ensemble"""
        try:
            return self.results["ensemble"]
        except:
            raise ValueError("Ensemble not calculated")

    def calculate(self, atoms=None, properties=["energy", "forces", "ensemble"], system_changes=all_changes):
        """
        Parameters
        ----------
        atoms : ase.Atoms
            ASE atoms object.
        properties : list of str
            energy, forces, and ensemble are supported.
        system_changes : list of str
            List of changes for ASE.
        """
        if atoms is not None:
            self.atoms = atoms.copy()

        model_inputs = self.ase_data_reader(self.atoms)

        if self.device == 'cuda':
            model_inputs = model_inputs.to(self.device)

        predictions = {'energy': [], 'forces': [],}
        for model in self.models:
            # Ensure model is in evaluation mode for deterministic inference
            model.eval()
            model_results = model(model_inputs)
            predictions['energy'].append(model_results.energy[0].detach().cpu().numpy().item() * self.energy_scale)
            if self.compute_forces:
                predictions['forces'].append(model_results.forces.detach().cpu().numpy() * self.forces_scale)
        
        results = {"energy": np.mean(predictions['energy'])}
        ensemble = {
            'energy_var': np.var(predictions['energy']),
        }
        
        if self.compute_forces:
            results["forces"] = np.mean(np.stack(predictions['forces']), axis=0)
            ensemble['forces_var'] =  np.var(np.stack(predictions['forces']), axis=0),
            ensemble['forces_l2_var'] = np.var(np.linalg.norm(predictions['forces'], axis=2), axis=0),

        results['ensemble'] = ensemble
        self.results = results


class AtomicEnsembleCalculator(Calculator):
    """
    Atomic ensemble calculator for multiple models.
    """
    implemented_properties = ["energy", "forces", "ensemble"]

    def __init__(
        self,
        model_paths=None,
        models=None,
        config=None,
        energy_scale=1.0,
        forces_scale=1.0,
        device=None,
        verbose=False,
        **kwargs):
        """
        Parameters
        ----------
        model_paths : list of str
            Paths to the models (several models for ensemble).
        models : list of iann.models.Model
            Models to use.
        config : dict
            Configuration for the calculator.
        energy_scale : float
            Energy scale.
        forces_scale : float
            Forces scale.
        device : str
            Device to use.
        verbose : bool
            Verbosity.

        Returns
        -------
        Dict[str, array]
            Dictionary with keys 'energy', 'forces', 'ensemble' including atomic energy variance.
        """
        super().__init__(**kwargs)

        if config is not None:
            self.config = config
        else:
            self.config = {}

        if "compute_forces" in self.config:
            self.compute_forces = self.config["compute_forces"]
        else:
            if "compute_forces" in kwargs:
                self.compute_forces = kwargs["compute_forces"]
                kwargs.pop("compute_forces")
            else:
                self.compute_forces = None

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if model_paths is not None:
            self.models = [_load_model(model_path, self.device, self.compute_forces, **kwargs) for model_path in model_paths]
        elif models is not None:
            self.models = models
        else:
            raise ValueError("Either model_paths or models must be provided")

        self.cutoff = self.models[0].cutoff
        self.compute_forces = self.models[0].compute_forces
        self.ase_data_reader = AseDataReader(self.cutoff, self.compute_forces)
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale

        if verbose:
            print(f"Atomic ensemble calculator initialized with {len(self.models)} models")
            print(f"Cutoff: {self.cutoff}")
            print(f"Compute forces: {self.compute_forces}")
            print(f"Energy scale: {self.energy_scale}")
            print(f"Forces scale: {self.forces_scale}")
    
    def get_ensemble(self):
        """Get the calculated ensemble"""
        try:
            return self.results["ensemble"]
        except:
            raise ValueError("Ensemble not calculated")

    def calculate(self, atoms=None, properties=["energy", "forces", "ensemble"], system_changes=all_changes):
        """
        Parameters
        ----------
        atoms : ase.Atoms
            ASE atoms object.
        properties : list of str
            energy, forces, and ensemble are supported.
        system_changes : list of str
            List of changes for ASE.
        """
        if atoms is not None:
            self.atoms = atoms.copy()

        model_inputs = self.ase_data_reader(self.atoms)

        if self.device == 'cuda':
            model_inputs = model_inputs.to(self.device)

        predictions = {'energy': [], 'forces': [], 'atomic_energy': []}
        for model in self.models:
            # Ensure model is in evaluation mode for deterministic inference
            model.eval()
            model_results = model(model_inputs)
            predictions['energy'].append(model_results.energy[0].detach().cpu().numpy().item() * self.energy_scale)
            predictions['atomic_energy'].append(model_results.atomic_energy.detach().cpu().numpy() * self.energy_scale)
            if self.compute_forces:
                predictions['forces'].append(model_results.forces.detach().cpu().numpy() * self.forces_scale)
        
        results = {"energy": np.mean(predictions['energy'])}
        ensemble = {
            'energy_var': np.var(predictions['energy']),
            'atomic_energy_var': np.var(np.stack(predictions['atomic_energy']), axis=0),
        }
        
        if self.compute_forces:
            results["forces"] = np.mean(np.stack(predictions['forces']), axis=0)
            ensemble['forces_var'] =  np.var(np.stack(predictions['forces']), axis=0),
            ensemble['forces_l2_var'] = np.var(np.linalg.norm(predictions['forces'], axis=2), axis=0),

        results['ensemble'] = ensemble
        self.results = results