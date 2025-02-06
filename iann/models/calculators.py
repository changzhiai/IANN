from ase.calculators.calculator import Calculator, all_changes
from iann.data.data import AseDataReader
import numpy as np

class MLCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model,
        energy_scale=1.0,
        forces_scale=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = model
        self.model_device = next(model.parameters()).device
        self.cutoff = model.cutoff
        self.compute_forces = model.compute_forces
        self.ase_data_reader = AseDataReader(self.cutoff, self.compute_forces)
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
        

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
        model_inputs = {
            k: v.to(self.model_device) for (k, v) in model_inputs.items()
        }

        model_results = self.model(model_inputs)

        results = {}

        # Convert outputs to calculator format
        if self.compute_forces:
            results["forces"] = (
                model_results["forces"].detach().cpu().numpy() * self.forces_scale
            )
        results["energy"] = (
            model_results["energy"][0].detach().cpu().numpy().item()
            * self.energy_scale
        )

        if model_results.get("fps"):
            atoms.info["fps"] = model_results["fps"].detach().cpu().numpy()
    
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




