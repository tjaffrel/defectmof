# defectmof

A Python toolkit for studying defects in metal-organic frameworks. Given a defective and a pristine unit cell (as CIF files), defectmof builds mixed supercells with controlled spatial arrangements of defects, relaxes them with MACE machine-learning potentials, runs molecular dynamics, and computes pair distribution functions for comparison with X-ray total scattering experiments.

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[pdf]"        # diffpy-CMI + DebyeCalculator for PDF
pip install -e ".[torchsim]"   # GPU-batched optimization/MD via torch-sim
pip install -e ".[all]"        # everything
```

## Quick start

### Python

```python
import defectmof

sc = defectmof.build_supercell(
    "defective.cif", "pristine.cif",
    size=(4, 4, 4), defect_fraction=0.44, mode="random",
)

optimized = defectmof.optimize(sc, model="mace_mp_medium", device="cuda")

snapshots = defectmof.run_md(optimized, temperature=150.0, n_steps=5000)

r, G_r = defectmof.compute_pdf(snapshots, rmin=1.0, rmax=30.0)
```

### Command line

Every step has a Hydra-powered CLI. You can pass parameters inline:

```bash
defectmof-build defective=defective.cif pristine=pristine.cif \
    size=[4,4,4] defect_fraction=0.44 mode=random output=supercell.cif

defectmof-optimize input=supercell.cif model=mace_mp_medium \
    fmax=0.05 device=cuda output=optimized.cif

defectmof-md input=optimized.cif temperature=150 n_steps=10000 \
    model=mace_mp_medium trajectory=md.traj

defectmof-pdf trajectory=md.traj rmin=1.0 rmax=20.0 \
    scattering=xray engine=diffpy output=pdf.csv

defectmof-rdf trajectory=md.traj rmax=10.0 elements=[Al,O] output=rdf.csv
```

### YAML config files

Instead of passing every parameter on the command line, you can write a YAML file and point to it with `--config-dir` and `--config-name`. Example configs are provided in `examples/configs/`.

**Build a supercell** (`examples/configs/supercell.yaml`):

```yaml
defective: struct1_optimized_c_axis.cif
pristine: struct2_optimized_a_axis.cif
size: [4, 4, 4]
defect_fraction: 0.44
mode: random
seed: 42
output: supercell.cif
```

```bash
defectmof-build --config-dir=examples/configs --config-name=supercell
```

**Run MD** (`examples/configs/md_150K.yaml`):

```yaml
input: optimized.cif
temperature: 150.0
n_steps: 5000
timestep: 1.0
model: mace_mp_medium
thermostat: langevin
friction: 0.01
log_interval: 10
equilibration_steps: 1000
backend: ase
device: cuda
trajectory: md_150K.traj
```

```bash
defectmof-md --config-dir=examples/configs --config-name=md_150K
```

You can override any YAML parameter from the command line:

```bash
defectmof-md --config-dir=examples/configs --config-name=md_150K \
    temperature=300 n_steps=20000 model=models/mofs_v2.model
```

The full set of example configs:

| File | Command | What it does |
|------|---------|--------------|
| `supercell.yaml` | `defectmof-build` | Build a 4x4x4 supercell |
| `optimize.yaml` | `defectmof-optimize` | Geometry optimization with L-BFGS |
| `md_150K.yaml` | `defectmof-md` | MD at 150 K, 5 ps, Langevin |
| `pdf_from_md.yaml` | `defectmof-pdf` | X-ray PDF from trajectory |

## Supercell arrangement modes

`build_supercell` supports 10 spatial arrangements for placing defective vs pristine unit cells:

| Mode | Description |
|------|-------------|
| `random` | Random placement to meet the target fraction |
| `alternating_a` / `_b` / `_c` | Alternate along a single axis |
| `alternating_ab` / `_ac` / `_bc` | Checkerboard in a plane |
| `alternating_abc` | 3D checkerboard |
| `clustered_small` | Defects grouped in 2x2x2 blocks |
| `clustered_large` | Defects grouped in half-cell-sized blocks |

## MACE models

Any MACE model works — pass a foundation model name or a path to a local `.model` file:

| Model | Description | How to use |
|-------|-------------|------------|
| `mace_mp_medium` | Foundation model (auto-downloaded) | `model=mace_mp_medium` |
| MOFs v2 | MOF-specific fine-tuned model | `model=models/mofs_v2.model` |
| MOF-OMAT v2 | MOF-OMAT fine-tuned model | `model=models/mof-omat-0-v2.model` |

When using a fine-tuned model with multiple heads, pass the `head` parameter (e.g. `head=pt_head`).

## Included structures

Four CIF files ship with the repo in `examples/structures/`:

| File | Role |
|------|------|
| `struct1_optimized_c_axis.cif` | Defective unit cell (Br-substituted linker) |
| `struct2_optimized_a_axis.cif` | Pristine unit cell |
| `CONTCAR_final.cif` | VASP-optimized defective structure |
| `CONTCAR_optimized.cif` | VASP-optimized pristine structure |

## Project layout

```
src/defectmof/          Core library (supercell, optimize, md, pdf)
examples/structures/    Unit cell CIF files
examples/configs/       Example YAML configs for the CLI
models/                 Fine-tuned MACE .model files
analysis/               Plotting scripts and results
tests/                  Test suite
```

## Contributing

1. Create a feature branch: `git checkout -b feat/my-feature`
2. Add tests in `tests/`, run `pytest tests/ -v`
3. Lint: `ruff check src/ tests/`
4. Open a PR against `master`

## License

MIT
