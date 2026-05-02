import os
from iann.foundations import foundation_model, list_available_models
from iann.calculators import MLCalculator
from ase.build import fcc100

def test_foundation_models():
    models = list_available_models()
    print(f"Found {len(models)} foundation models: {models}")
    
    atoms = fcc100("Pt", size=(4,4,3), a=5.5, vacuum=15.0)
    
    success_count = 0
    for model_name in models:
        print(f"\n--- Testing Foundation Model: {model_name} ---")
        try:
            # Create calculator with the foundation model
            calc = MLCalculator(
                model_path=foundation_model(model_name),
                compute_forces=True,
                device='cpu'
            )
            
            # Set the calculator
            atoms.calc = calc
            
            # Get predictions
            energy = atoms.get_potential_energy()
            print(f"SUCCESS: {model_name} Energy: {energy:.4f} eV")
            success_count += 1
            
        except Exception as e:
            print(f"FAILED: {model_name} - Error: {e}")
    
    print(f"\n--- Foundation Model Testing Summary: {success_count}/{len(models)} passed ---")
    if success_count < len(models):
        exit(1)

if __name__ == "__main__":
    test_foundation_models()