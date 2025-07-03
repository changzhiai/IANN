from ase.calculators.calculator import Calculator, all_changes
from iann.data.data import AseDataReader
import numpy as np
import torch


class MLCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_path=None,
        model=None,
        config=None,
        energy_scale=1.0,
        forces_scale=1.0,
        device=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            raise ValueError("Either model or model_path must be provided")
        
        if config is not None:
            self.config = config

        self.model_device = next(self.model.parameters()).device
        self.cutoff = self.model.cutoff
        self.compute_forces = self.model.compute_forces
        self.ase_data_reader = AseDataReader(self.cutoff, self.compute_forces)
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
        

    def _load_model(self, model_path):
        """Load model from path and determine its type."""
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Determine model type from state dict
        if "model_type" in state_dict:
            model_type = state_dict["model_type"]
        else:
            # Try to determine from model architecture
            if "num_layer" in state_dict:
                model_type = "painn"
            elif "irreps" in state_dict:
                model_type = "nequip"
            elif "correlation" in state_dict:
                model_type = "mace"
            elif "transformer" in state_dict:
                model_type = "equiformerV2"
            else:
                raise ValueError("Could not determine model type from state dict")

        # Create appropriate model
        if model_type == "painn":
            from iann.models.painn import PaiNN
            model = PaiNN(
                num_interactions=state_dict["num_layer"],
                hidden_state_size=state_dict["node_size"],
                cutoff=state_dict["cutoff"],
                compute_forces=True
            )
        elif model_type == "nequip":
            from iann.models.nequip import NequIP
            model = NequIP(
                num_interactions=state_dict["num_layer"],
                num_features=state_dict["node_size"],
                cutoff=state_dict["cutoff"],
                compute_forces=True
            )
        elif model_type == "mace":
            from iann.models.mace import MACE
            model = MACE(
                num_interactions=state_dict["num_layer"],
                num_features=state_dict["node_size"],
                correlation = 3,
                species = None,
                cutoff=state_dict["cutoff"],
                compute_forces=True,
            )
        elif model_type == "equiformerV2":
            from iann.models.equiformerV2 import EquiformerV2
            model = EquiformerV2(
                num_interactions=state_dict["num_layer"],
                num_features=state_dict["node_size"],
                cutoff=state_dict["cutoff"],
                compute_forces=True
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Please choose from: painn, nequip, mace, and equiformerV2!")
        
        model.to(self.device)
        # Load state dict
        model.load_state_dict(state_dict["model"])
        return model

    def calculate(self, atoms=None, properties=["energy",], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        if atoms is not None:
            self.atoms = atoms.copy()       

        model_inputs = self.ase_data_reader(self.atoms)
        
        if self.device == 'cuda':
            model_inputs = model_inputs.to(self.device)

        model_results = self.model(model_inputs)

        results = {}
        results["energy"] = model_results.energy[0].detach().cpu().numpy().item()
        if self.compute_forces:
            results["forces"] = model_results.forces.detach().cpu().numpy() * self.forces_scale
    
        self.results = results


class EnsembleCalculator(Calculator):
    implemented_properties = ["energy", "forces", "ensemble"]

    def __init__(
        self,
        models,
        energy_scale=1.0,
        forces_scale=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.models = models
        self.model_device = next(models[0].parameters()).device
        self.cutoff = models[0].cutoff
        self.compute_forces = models[0].compute_forces
        self.ase_data_reader = AseDataReader(self.cutoff, self.compute_forces)
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
    
    def get_ensemble(self):
        """Get the calculated ensemble"""
        try:
            return self.results["ensemble"]
        except:
            return None

    def calculate(self, atoms=None, properties=["energy", "forces", "ensemble"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        if atoms is not None:
            self.atoms = atoms.copy()       

        model_inputs = self.ase_data_reader(self.atoms)
        model_inputs = {
            k: v.to(self.model_device) for (k, v) in model_inputs.items()
        }

        predictions = {'energy': [], 'forces': [], 'atomic_energy': []}
        for model in self.models:
            model_results = model(model_inputs)
            predictions['energy'].append(model_results["energy"][0].detach().cpu().numpy().item() * self.energy_scale)
            predictions['atomic_energy'].append(model_results["atomic_energy"].detach().cpu().numpy() * self.energy_scale)
            if bool(self.compute_forces):
                predictions['forces'].append(model_results["forces"].detach().cpu().numpy() * self.forces_scale)
        
        results = {"energy": np.mean(predictions['energy'])}
        ensemble = {
            'energy_var': np.var(predictions['energy']),
            'atomic_energy_var': np.var(np.stack(predictions['atomic_energy']), axis=0),
        }
        
        if bool(self.compute_forces):
            results["forces"] = np.mean(np.stack(predictions['forces']), axis=0)
            ensemble['forces_var'] =  np.var(np.stack(predictions['forces']), axis=0),
            ensemble['forces_l2_var'] = np.var(np.linalg.norm(predictions['forces'], axis=2), axis=0),

        results['ensemble'] = ensemble
        self.results = results


class AtomicEnsembleCalculator(Calculator):
    implemented_properties = ["energy", "forces", "ensemble"]

    def __init__(
        self,
        models,
        energy_scale=1.0,
        forces_scale=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.models = models
        self.model_device = next(models[0].parameters()).device
        self.cutoff = models[0].cutoff
        self.compute_forces = models[0].compute_forces
        self.ase_data_reader = AseDataReader(self.cutoff, self.compute_forces)
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
    
    def get_ensemble(self):
        """Get the calculated ensemble"""
        try:
            return self.results["ensemble"]
        except:
            return None

    def calculate(self, atoms=None, properties=["energy", "forces", "ensemble"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        if atoms is not None:
            self.atoms = atoms.copy()

        model_inputs = self.ase_data_reader(self.atoms)
        model_inputs = {
            k: v.to(self.model_device) for (k, v) in model_inputs.items()
        }

        predictions = {'energy': [], 'forces': []}
        for model in self.models:
            model_results = model(model_inputs)
            predictions['energy'].append(model_results["energy"][0].detach().cpu().numpy().item() * self.energy_scale)
            predictions['forces'].append(model_results["forces"].detach().cpu().numpy() * self.forces_scale)

        results = {"energy": np.mean(predictions['energy'])}
        results["forces"] = np.mean(np.stack(predictions['forces']), axis=0)

        ensemble = {
            'energy_var': np.var(predictions['energy']),
            'forces_var': np.var(np.stack(predictions['forces']), axis=0),
            'forces_l2_var': np.var(np.linalg.norm(predictions['forces'], axis=2), axis=0),
        }

        results['ensemble'] = ensemble

        self.results = results




