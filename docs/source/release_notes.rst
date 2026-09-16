Release Notes
=============

Version numbers follow the tags in the `GitHub repository
<https://github.com/changzhiai/IANN/tags>`_. Each section lists what changed
relative to the release before it.

Development started on 7 November 2024; the first tag came a little over a year
later, so most of the early history predates any release boundary and is
summarised under :ref:`0.0.0 (2024-11-07) <release-0.0.0>`.

.. _release-0.1.2:

0.1.2 (unreleased)
------------------

In development on the ``master`` branch. The headline change is that the set of
architectures grew from four to eight.

Added
~~~~~

* **Three new architectures.** Allegro, EquiformerV3 and UMA join PaiNN,
  NequIP, MACE and EquiformerV2. All are selected the same way — by
  name in the configuration dictionary — and are documented in
  :doc:`engine_models`.
* **Foundation models at three DFT levels.** Twelve pretrained PaiNN checkpoints
  spanning PBE, RPBE and r\ :sup:`2`\ SCAN, trained on 41.9 million curated
  structures, plus seven architecture-comparison checkpoints. Naming a model
  downloads and caches it; see :doc:`foundation_models`.
* **Stress and virial** as trained and predicted quantities, obtained from the
  same per-edge gradients as the forces, across all architectures.
* **Per-atom energy scaling and shifting** per element type, and a per-atom loss
  option.
* ``reset_lr``, which starts a fresh learning-rate schedule when loading a
  checkpoint instead of resuming the previous one — the usual requirement when
  fine-tuning a foundation model onto new chemistry.
* ``model_config`` **in the checkpoint**, so architecture hyperparameters are
  persisted with the weights and calculators can rebuild a model without being
  told how it was configured.
* **Configurable schedulers**, including ``CosineAnnealingWarmRestarts`` with
  ``T_0`` / ``T_mult``, and learning-rate adjustment on restart.
* **Average-neighbour-count computation** in the trainer, with normalisation
  statistics that can be estimated from a subsample and read in parallel.
* **LAMMPS export for Allegro, EquiformerV3 and UMA.** The export path
  previously reached only four of the architectures, so a model trained with
  one of the other three could not be deployed. All seven now export, and every
  change was gated on the exported model reproducing the eager energy and
  forces, so existing checkpoints continue to load unchanged.
* **An agent-facing interface**, as the ``iann/agent/`` subpackage: the ``iann``
  console command with ``--json`` on every subcommand, an MCP server, tool-neutral
  repository notes, and five Claude Code skills — the last two installed by
  ``iann agent install``. Operations are reads plus explicitly bounded training.
  See :doc:`agents`.
* ``iann doctor``, which probes each dependency separately and separates a hard
  import failure from a silent degradation.
* ``iann inspect``, which reports the structural parameters a checkpoint does
  **not** record — the cause of size mismatches when rebuilding EquiformerV3 or
  UMA models.
* **Documentation:** :doc:`about`, :doc:`performance`, :doc:`agents`, an
  expanded :doc:`engine_models` with architecture diagrams, and a project logo.

Changed
~~~~~~~

* ``use_cuequivariance`` was renamed to ``use_cue`` across all models.
* Allegro was rewritten to the full architecture of the paper rather than a
  reduced form.
* Hyperparameter extraction was centralised, and the TorchScript export and
  model-saving paths were reworked around it.
* Tensor dimensions for ``cell``, ``virial`` and ``stress`` were standardised.
* NequIP gained MACE-style residual readouts and a fixed 119-element embedding
  for cross-dataset compatibility.

Fixed
~~~~~

* ``ReduceLROnPlateau`` was not being stepped correctly, so the learning rate
  never dropped.
* Distributed training under **Torque/PBS** failed to start.
* SLURM nodelist parsing mis-reported the node actually in use.
* Forces in EquiformerV2, and a neighbour tensor-product error.
* ``retain_graph`` / ``create_graph`` now follow ``self.training`` rather than
  being always on, in PaiNN, FastPot and Demo.
* TorchScript export with cuEquivariance enabled.
* **EquiformerV2 could no longer be exported to TorchScript.** The change that
  reworked its forces introduced a call TorchScript cannot compile, which went
  unnoticed because the export check aborted on the first architecture that
  failed. That check now tries each architecture independently and reports a
  summary.
* ``dtype`` and irreps assignment errors in MACE.

0.1.1 (2026-04-27)
------------------

Added
~~~~~

* **cuEquivariance support for MACE and NequIP**, switchable at runtime, for
  NVIDIA's optimised equivariant kernels.
* A foundation model trained on Materials Project data (MPtrj), and support for
  more than one bundled foundation model.
* Troubleshooting guidance for C++ compilation errors during dependency
  installation.

Fixed
~~~~~

* Inconsistent tensor shapes in ``index_add_`` for NequIP and MACE.
* Unintended output irreps in MACE, by constraining ``filter_irreps_out``.
* Single-atom systems, which previously failed.
* Case sensitivity in the ``painn_mptrj.pt`` filename.

0.1.0 (2025-12-01)
------------------

First tagged release, closing about thirteen months and 405 commits of
development that began on **7 November 2024**.

It contains PaiNN, NequIP, MACE and EquiformerV2 behind one data object and one
trainer, unified ASE calculators, ensemble uncertainties, and the LAMMPS pair
styles.

Archived at `doi:10.5281/zenodo.17809949 <https://doi.org/10.5281/zenodo.17809949>`_.



.. _release-0.0.0:

0.0.0 (2024-11-07)
------------------

First commit. The initial version of the framework — implemented the first architecture with user-friendly configuration.

The repository has no tags before ``v0.1.0``, so this period has no release
boundaries to list. The order in which the framework came together:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Date
     - Milestone
   * - 2024-11-07
     - First commit.
   * - 2025-05-06
     - The framework proper — the unified trainer, the calculators,
       EquiformerV2, and the C++ LAMMPS pair style.
   * - 2025-10-27
     - Foundation-model loading.
   * - 2025-12-01
     - Tagged ``v0.1.0`` and archived on Zenodo.