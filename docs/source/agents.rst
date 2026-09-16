Agent interface
===============

IANN ships a machine-readable interface so that an automated caller (agent interface), such as a CI job, a script, or an AI coding skills can be directly used by LLM agents. The agent interface comes in three parts: **Command line**, **MCP server**, and **Agent skills**.


Everything in this part lives in the ``iann/agent/`` subpackage, so it can be read, reviewed or removed as a
unit. Nothing in it is imported by the rest of IANN.


Command line
------------

Installed as a console script with the package:

.. code-block:: bash

   pip install -e .
   iann --help

Add ``--json`` to any subcommand to get the raw result instead of a summary. Under ``--json``,
**stdout carries nothing but the JSON document** — the library's own progress output is redirected
to stderr, where it stays visible without corrupting what a caller parses.

Exit codes are meaningful, which is what makes the command usable in a check:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Code
     - Meaning
   * - ``0``
     - the operation succeeded
   * - ``1``
     - the operation failed; a ``{"error": ..., "message": ...}`` object goes to stderr
   * - ``2``
     - the command line was wrong

A failure is never a bare traceback and never a zero exit with an empty result. ``iann export`` in
particular **raises** where :func:`~iann.plugins.converter.convert_model_for_lammps` returns
``None`` and prints, because a caller would otherwise read that silence as success.

Subcommands
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Command
     - What it returns
   * - ``iann doctor``
     - Whether this environment can run IANN: the interpreter, each dependency probed separately,
       CUDA availability, and which architectures actually construct.
   * - ``iann models``
     - The architectures the trainer accepts, which reach LAMMPS, and the structural parameters
       each one needs that checkpoints do not record.
   * - ``iann foundation list [--group painn|arch|all]``
     - The released foundation models with their energy and force MAEs.
   * - ``iann foundation info NAME``
     - One model in detail, including whether it is already cached and where.
   * - ``iann foundation fetch NAME [--local-files-only]``
     - A local path, downloading through ``huggingface_hub`` if needed.
   * - ``iann inspect CKPT``
     - What a checkpoint records — architecture, step, best validation loss, channel and layer
       counts — and a ``missing_config`` list of what it does **not**.
   * - ``iann predict --model CKPT --structure FILE [--index N] [--ensemble CKPT ...]``
     - Single-point energy and forces; with ``--ensemble``, also the variance across the models.
   * - ``iann train --model ARCH --dataset PATH --max-steps N [--out DIR]``
     - A bounded run, plus a ``run.json`` manifest beside the checkpoint recording the config used,
       the steps run and the final metrics.
   * - ``iann status --output-dir DIR``
     - Progress of a run from its log and its checkpoint. Safe to call while training is still
       going.
   * - ``iann export --model CKPT [--type ARCH] [--out PATH]``
     - A TorchScript file for LAMMPS ``pair_style iann``. The architecture is inferred from the
       checkpoint when ``--type`` is omitted.
   * - ``iann agent install [--target DIR] [--force]``
     - Writes ``AGENTS.md``, ``CLAUDE.md`` and ``.claude/skills/`` into a repository.

Check environment 
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   iann doctor

``doctor`` exits non-zero on a broken environment, and it distinguishes a hard failure — ``asap3``
cannot import at all — from a silent degradation, where ``torch`` imports but warns that NumPy
failed to initialise. That second case is the one worth having a tool for: it produces confusing
failures much later, in unrelated code. It also imports nothing at module scope, so it still runs
in an environment too broken to ``import iann``.

Inspect checkpoint
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   iann inspect test/painn/output/model.pt

Checkpoints are not fully self-describing. ``num_distance_basis`` and the grid-resolution lists are
never persisted, so rebuilding an EquiformerV3 or UMA model needs the original training keyword
arguments or ``load_state_dict`` fails with size mismatches. ``inspect`` reports the gap in its
``missing_config`` field; pass the missing values back through ``--config``.

Test cli
~~~~~~~~

.. code-block:: bash

   python iann/agent/tests/test_cli.py           # every subcommand, plus the failure paths
   python iann/agent/tests/test_cli.py --fast    # skip the bounded training run

A plain script in the style of the rest of ``test/``, exiting 0 on success and 1 on failure. It
runs a real end-to-end loop — ``train --max-steps 2`` → ``status`` → ``inspect`` → ``predict`` →
``export`` — and asserts each promised artefact exists.

MCP server
----------

The server exposes nine tools over the Model Context Protocol: ``iann_doctor``,
``iann_list_architectures``, ``iann_list_foundation_models``, ``iann_foundation_info``,
``iann_inspect_checkpoint``, ``iann_predict``, ``iann_train``, ``iann_training_status`` and
``iann_export_lammps``.

It needs the optional MCP SDK:

.. code-block:: bash

   pip install -e ".[agent]"
   python -m iann.agent.mcp_server

Both generations of the SDK are supported — 2.x moved the supported server surface from the
low-level ``Server`` decorators to ``MCPServer`` — so ``pip`` resolving to whatever is current does
not break the server.

Register MCP
~~~~~~~~~~~~
Register it with Claude Code, using the interpreter of the environment IANN is installed in:

.. code-block:: bash

   claude mcp add iann -- /opt/anaconda3/envs/iann/bin/python -m iann.agent.mcp_server

The ``iann`` command works without the SDK; only the server needs it, and it says so rather than
failing obscurely when the SDK is absent.

.. note::

   The MCP SDK pulls in ``cryptography``, which ships a compiled extension. In a conda environment
   whose OpenSSL is older than the wheel expects, importing it fails with a missing symbol such as
   ``symbol not found in flat namespace '_EVP_DigestSqueeze'``. It is the dependency that is broken,
   not the server; installing a slightly older ``cryptography`` resolves it.

Test MCP
~~~~~~~~

.. code-block:: bash

   python iann/agent/tests/test_mcp.py

Checks the tool table without needing the SDK — that part is shared with the command line, so it is
what breaks when :mod:`iann.agent.commands` changes — then starts the server over stdio and calls
each read-only tool. The stdio round trip is skipped rather than failed when the SDK is absent.

Agent skills
------------

``iann/agent/AGENTS.md`` carries the repository knowledge an agent cannot infer from the source:
which conda environment works, that scripts must run from the repository root, that every training
run needs a step budget, and that checkpoints are not self-describing. **Nothing in it is specific
to one assistant** — it is written for any AI coding agent, and it is worth a human read too.

The *filenames* agents look for are not neutral, though. ``AGENTS.md`` is the cross-tool
convention, while Claude Code reads only ``CLAUDE.md``. So the one source is written out under both
names:

.. code-block:: bash

   iann agent install

This writes ``AGENTS.md`` and ``CLAUDE.md`` at the repository root, with identical content, plus
``.claude/skills/<name>/SKILL.md`` for each of the five skills. Skills *are* a Claude Code feature,
so those have only the one destination. The command is idempotent, and it will not overwrite a file
you have edited without ``--force``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Skill
     - Covers
   * - ``iann-train``
     - Choosing an architecture, bounding the run, watching progress, reading the manifest.
   * - ``iann-export-lammps``
     - The repeat-the-config trap, the per-architecture keyword arguments, the ``pair_style``
       strings, and that FastPot cannot be exported while UMA needs ``num_experts == 0``.
   * - ``iann-foundation-models``
     - The catalog, fetching, fine-tuning with ``reset_lr``, and working offline on compute nodes.
   * - ``iann-hpc-submit``
     - Generating a SLURM or PBS submission script — and stopping there.
   * - ``iann-run-checks``
     - Running the test suite, the docs build and the LAMMPS export check, and reading each one's
       output correctly.

That last skill exists because two of the test runner's statuses do not mean what they look like:
``VERIFIED`` means the task hit the harness timeout rather than that it converged, and pass/fail is
decided by scanning child output for substrings that some children legitimately print in their own
summary tables.
