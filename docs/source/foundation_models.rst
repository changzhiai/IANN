Foundation Models
=================

Foundation models available
---------------------------

Currently, we provide the following foundation models:

- ``painn_oc.pt``: Trained on the **OpenCatalysis** dataset (OC20 and OC22), containing 10,000,000+ structures.
- ``painn_mptrj.pt``: Trained on the **Materials Project** trajectory dataset (MPtrj), containing 1,500,000+ structures.

These models are intended to accelerate downstream tasks such as property prediction or molecular dynamics by leveraging knowledge learned from large-scale datasets.

For example, you can use the following code to load a foundation model:

.. code-block:: python

   from iann.foundations import foundation_model
   from iann.calculators import MLCalculator
   from ase.build import fcc100

   calc = MLCalculator(
      model_path=foundation_model("painn_oc.pt"), # foundation model trained on OC20+OC22
      compute_forces=True,
      device='cpu') # use 'cuda' for GPU

   atoms = fcc100("Pt", size=(4,4,3), a=5.5, vacuum=15.0)
   atoms.calc = calc
   nnp_energy = atoms.get_potential_energy()
   nnp_forces = atoms.get_forces()
   print(f"NNP Energy: {nnp_energy:.4f} eV")
   print(f"NNP Forces: {nnp_forces}")

And you can fine tune the foundation model on your own dataset by using the following code:

.. code-block:: python

   from iann.trainer import Trainer
   from iann.foundations import foundation_model

   trainer = Trainer(model="painn", 
        config={"num_channels": 128, # number of channels in the model
            "num_layers": 3, # number of layers in the model
            "cutoff": 5.5, # cutoff radius
            "batch_size": 16, # batch size
            "learning_rate": 0.0001, # initial learning rate
            "forces_weight": 0.9, # weight for forces
            "load_model": foundation_model("painn_oc.pt"), # load the foundation model
            "max_steps": 10000000, # maximum number of steps
            "random_seed": 888, # random seed for reproducibility
            "val_ratio": 0.003, # validation ratio
            "stop_patience": 500, # patience for early stopping
            'device': 'cuda',
            'output_dir': 'output',
            'output_log': 'output.log',
            'output_model': 'model.pt'},
        distributed=False)
   trainer.train("dataset.traj")


Foundation models in future
---------------------------

The more comprehensive the training data is, the better the foundation model will perform. The more comprehensive foundation models will be provided in future releases. The more training data could be sourced from well-established materials databases, such as:

- `Catalysis-Hub (CatHub) <https://www.catalysis-hub.org>`_ — a repository of surface reaction data.
- `Materials Project <https://materialsproject.org>`_ — a comprehensive database of computed materials properties.
- `OpenCatalysis <https://opencatalystproject.org/>`_ — a large-scale dataset for heterogeneous catalysis research.

These models will be compatible with the rest of the IANN framework and can be fine-tuned or used out-of-the-box for various atomistic simulations.

