# defectmof

**Simple tools for MOF defect supercell generation, optimization, and simulation.**

defectmof is a Python library designed for experimentalists studying defects in metal-organic frameworks (MOFs). Given a defective unit cell and a pristine unit cell as CIF files, defectmof builds mixed supercells with controlled spatial distributions of defects, optimizes them with machine-learning force fields (MACE), runs molecular dynamics, and computes pair distribution functions for direct comparison with X-ray total scattering experiments.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [API Reference](#api-reference)
4. [CLI Reference](#cli-reference)
5. [Hydra Config Reference](#hydra-config-reference)
6. [Examples](#examples)
7. [Included Structures](#included-structures)
8. [Benchmark Reference](#benchmark-reference)
9. [Contributing](#contributing)

---

## Installation

### pixi (recommended)

```bash
pixi add defectmof
```

### pip

```bash
pip install defectmof
```

### Optional dependencies

Some features require optional packages that are not installed by default.

| Extra | What it enables | Install command |
|-------|----------------|-----------------|
| `torchsim` | GPU-batched optimization and MD via torch-sim | `pip install "defectmof[torchsim]"` |
| `pdf` | PDF computation via diffpy-CMI (`diffpy` engine) and DebyeCalculator (`debye_calculator` engine) | `pip install "defectmof[pdf]"` |
| `dev` | pytest and ruff for running the test suite | `pip install "defectmof[dev]"` |
| `all` | All optional dependencies | `pip install "defectmof[all]"` |

The `torchsim` package is installed from GitHub:

```bash
pip install git+https://github.com/TorchSim/torch-sim
```

The `pdf` extras install `diffpy.srreal` and `debyecalculator`:

```bash
pip install diffpy.srreal debyecalculator
```

---

## Quick Start

### Python API

```python
import defectmof

# Build a 4x4x4 supercell with 44% defective cells, random arrangement
sc = defectmof.build_supercell("defective.cif", "pristine.cif",
                               size=(4, 4, 4), defect_fraction=0.44)

# Optimize with MACE on GPU
optimized = defectmof.optimize(sc, model="mace_mp_medium", device="cuda")

# Run MD at 150 K for 5000 steps
snapshots = defectmof.run_md(optimized, temperature=150.0, n_steps=5000)

# Compute pair distribution function
r, G_r = defectmof.compute_pdf(snapshots, rmin=1.0, rmax=30.0)
```

### YAML config (Hydra CLI)

```yaml
# configs/run.yaml
defective: examples/structures/struct1_optimized_c_axis.cif
pristine:  examples/structures/struct2_optimized_a_axis.cif
size:      [4, 4, 4]
defect_fraction: 0.44
mode:      random
seed:      42
output:    supercell.cif
```

```bash
defectmof-build --config-dir=configs --config-name=run
```

---

## API Reference

### `build_supercell`

Build a mixed defective/pristine MOF supercell from two unit cells.

```python
defectmof.build_supercell(
    defective,
    pristine,
    size=(4, 4, 4),
    defect_fraction=0.44,
    mode="random",
    seed=42,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `defective` | `str \| Atoms` | required | Path to CIF file or ASE Atoms for the defective unit cell |
| `pristine` | `str \| Atoms` | required | Path to CIF file or ASE Atoms for the pristine unit cell |
| `size` | `tuple[int, int, int]` | `(4, 4, 4)` | Supercell repeat dimensions (nx, ny, nz) |
| `defect_fraction` | `float` | `0.44` | Fraction of unit cells placed as defective (0.0 to 1.0) |
| `mode` | `str` | `"random"` | Spatial arrangement mode (see table below) |
| `seed` | `int` | `42` | Random seed for reproducible arrangements |

**Returns:** `ase.Atoms` — the assembled supercell with overlapping atoms pruned and short C-Br bonds corrected.

**Raises:** `ValueError` if cell parameters do not match within tolerance, or if `mode` is not recognized. `FileNotFoundError` if a CIF path does not exist.

**Arrangement modes:**

| Mode | Description |
|------|-------------|
| `random` | Randomly assign defective/pristine cells to meet the target fraction |
| `alternating_a` | Alternate along the a-axis: defective on even, pristine on odd |
| `alternating_b` | Alternate along the b-axis |
| `alternating_c` | Alternate along the c-axis |
| `alternating_ab` | Checkerboard pattern in the ab-plane |
| `alternating_ac` | Checkerboard pattern in the ac-plane |
| `alternating_bc` | Checkerboard pattern in the bc-plane |
| `alternating_abc` | 3D checkerboard (alternating in all three directions) |
| `clustered_small` | Group defects into small 2×2×2 blocks |
| `clustered_large` | Group defects into large half-cell-sized blocks |

---

### `optimize`

Optimize a structure to a force convergence criterion using MACE.

```python
defectmof.optimize(
    atoms,
    model="mace_mp_medium",
    optimizer="lbfgs",
    fmax=0.05,
    max_steps=500,
    backend="ase",
    device="cuda",
    memory_limit_gb=None,
    output=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `atoms` | `str \| Atoms` | required | Path to CIF file or ASE Atoms object |
| `model` | `str` | `"mace_mp_medium"` | MACE model: `"mace_mp_small"`, `"mace_mp_medium"`, or a local path |
| `optimizer` | `str` | `"lbfgs"` | Optimization algorithm: `"lbfgs"`, `"fire"`, or `"bfgs"` |
| `fmax` | `float` | `0.05` | Force convergence criterion in eV/Å |
| `max_steps` | `int` | `500` | Maximum number of optimization steps |
| `backend` | `str` | `"ase"` | Computation backend: `"ase"` or `"torchsim"` |
| `device` | `str` | `"cuda"` | Compute device: `"cuda"` or `"cpu"`. Falls back to CPU if CUDA is unavailable |
| `memory_limit_gb` | `float \| None` | `None` | Maximum GPU memory for the torchsim backend. Ignored for ASE backend |
| `output` | `str \| None` | `None` | If set, save the optimized structure to this CIF path |

**Returns:** `ase.Atoms` — the optimized structure.

**Raises:** `ValueError` for unknown optimizer or backend. Issues a `UserWarning` if optimization does not converge within `max_steps`. Raises `RuntimeError` on GPU out-of-memory.

---

### `run_md`

Run a molecular dynamics simulation and collect snapshots.

```python
defectmof.run_md(
    atoms,
    temperature=150.0,
    n_steps=5000,
    timestep=1.0,
    model="mace_mp_medium",
    thermostat="langevin",
    friction=0.01,
    log_interval=10,
    equilibration_steps=1000,
    backend="ase",
    device="cuda",
    memory_limit_gb=None,
    trajectory=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `atoms` | `str \| Atoms` | required | Path to CIF file or ASE Atoms object |
| `temperature` | `float` | `150.0` | Simulation temperature in Kelvin |
| `n_steps` | `int` | `5000` | Number of production MD steps (after equilibration) |
| `timestep` | `float` | `1.0` | Timestep in femtoseconds |
| `model` | `str` | `"mace_mp_medium"` | MACE model name or local path |
| `thermostat` | `str` | `"langevin"` | Thermostat algorithm: `"langevin"` or `"nose_hoover"` |
| `friction` | `float` | `0.01` | Langevin friction coefficient in 1/fs. Ignored when using `"nose_hoover"` |
| `log_interval` | `int` | `10` | Save one snapshot every N production steps |
| `equilibration_steps` | `int` | `1000` | Number of equilibration steps to discard before collecting snapshots |
| `backend` | `str` | `"ase"` | Computation backend: `"ase"` or `"torchsim"` |
| `device` | `str` | `"cuda"` | Compute device: `"cuda"` or `"cpu"` |
| `memory_limit_gb` | `float \| None` | `None` | Maximum GPU memory for torchsim backend |
| `trajectory` | `str \| None` | `None` | If set, save the full trajectory to this file path |

**Returns:** `list[ase.Atoms]` — snapshots collected from the production phase only.

---

### `compute_pdf`

Compute the Pair Distribution Function G(r) averaged over MD snapshots.

```python
defectmof.compute_pdf(
    snapshots,
    rmin=1.0,
    rmax=30.0,
    rstep=0.01,
    scattering="xray",
    engine="diffpy",
    device="cuda",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `snapshots` | `list[Atoms] \| str` | required | List of ASE Atoms snapshots or path to a trajectory file |
| `rmin` | `float` | `1.0` | Minimum r in Angstroms |
| `rmax` | `float` | `30.0` | Maximum r in Angstroms |
| `rstep` | `float` | `0.01` | Step size for the r grid in Angstroms |
| `scattering` | `str` | `"xray"` | Scattering type: `"xray"` or `"neutron"`. Only used by the `diffpy` engine |
| `engine` | `str` | `"diffpy"` | PDF engine: `"diffpy"` (diffpy-CMI, CPU) or `"debye_calculator"` (GPU-accelerated) |
| `device` | `str` | `"cuda"` | Compute device for the `debye_calculator` engine |

**Returns:** `tuple[np.ndarray, np.ndarray]` — `(r, G_r)` arrays of distances and frame-averaged PDF.

**Notes:** The `diffpy` engine requires `pip install diffpy.srreal`. The `debye_calculator` engine requires `pip install debyecalculator`.

---

### `compute_rdf`

Compute the radial distribution function g(r) using ASE. This is a fast, unweighted RDF suitable for structural analysis. It is not directly comparable to experimental PDF data; use `compute_pdf` for that purpose.

```python
defectmof.compute_rdf(
    snapshots,
    rmax=10.0,
    nbins=200,
    elements=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `snapshots` | `list[Atoms] \| str` | required | List of ASE Atoms snapshots or path to a trajectory file |
| `rmax` | `float` | `10.0` | Maximum distance in Angstroms |
| `nbins` | `int` | `200` | Number of bins |
| `elements` | `tuple[str, str] \| None` | `None` | Element pair for a partial RDF, e.g. `("Al", "O")`. If `None`, computes the total RDF |

**Returns:** `tuple[np.ndarray, np.ndarray]` — `(r, g_r)` arrays.

---

### `hierarchical_optimize`

Optimize a large supercell in two stages to reduce computational cost: first build and optimize a 2×2×2 cell, then tile it to the target size and re-optimize.

```python
defectmof.hierarchical_optimize(
    defective,
    pristine,
    target_size=(8, 8, 8),
    defect_fraction=0.44,
    mode="random",
    model="mace_mp_medium",
    backend="ase",
    device="cuda",
    output=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `defective` | `str \| Atoms` | required | Path to CIF or ASE Atoms for the defective unit cell |
| `pristine` | `str \| Atoms` | required | Path to CIF or ASE Atoms for the pristine unit cell |
| `target_size` | `tuple[int, int, int]` | `(8, 8, 8)` | Final supercell dimensions. Each dimension must be >= 4 and even |
| `defect_fraction` | `float` | `0.44` | Fraction of defective unit cells |
| `mode` | `str` | `"random"` | Spatial arrangement mode (same options as `build_supercell`) |
| `model` | `str` | `"mace_mp_medium"` | MACE model name or local path |
| `backend` | `str` | `"ase"` | Computation backend: `"ase"` or `"torchsim"` |
| `device` | `str` | `"cuda"` | Compute device: `"cuda"` or `"cpu"` |
| `output` | `str \| None` | `None` | If set, save the final optimized structure to this CIF path |

**Returns:** `ase.Atoms` — the optimized structure at `target_size`.

**Raises:** `ValueError` if any dimension of `target_size` is less than 4 or is not divisible by 2.

**Stages:**
1. Build a 2×2×2 supercell with `defect_fraction` and `mode`, optimize it with `fmax=0.1`.
2. Tile the optimized 2×2×2 by `(target_size // 2)` to reach `target_size`, then re-optimize with `fmax=0.05`.

---

## CLI Reference

All commands use [Hydra](https://hydra.cc/) for configuration. You can pass parameters on the command line or point to a YAML config file.

### `defectmof-build` — Build a supercell

```bash
defectmof-build --config-dir=configs --config-name=supercell
# or pass parameters directly:
defectmof-build defective=defective.cif pristine=pristine.cif \
    size=[4,4,4] defect_fraction=0.44 mode=random output=supercell.cif
```

### `defectmof-optimize` — Optimize a structure

```bash
defectmof-optimize --config-dir=configs --config-name=optimize
# or:
defectmof-optimize input=supercell.cif model=mace_mp_medium \
    optimizer=lbfgs fmax=0.05 backend=ase device=cuda output=optimized.cif
```

### `defectmof-md` — Run molecular dynamics

```bash
defectmof-md --config-dir=configs --config-name=md
# or:
defectmof-md input=optimized.cif temperature=150 n_steps=5000 \
    thermostat=langevin trajectory=traj.traj
```

### `defectmof-pdf` — Compute PDF from trajectory

```bash
defectmof-pdf --config-dir=configs --config-name=pdf
# or:
defectmof-pdf trajectory=traj.traj rmin=1.0 rmax=30.0 \
    engine=diffpy scattering=xray output=pdf_results.csv
```

### `defectmof-rdf` — Compute RDF from trajectory

```bash
defectmof-rdf --config-dir=configs --config-name=rdf
# or:
defectmof-rdf trajectory=traj.traj rmax=10.0 nbins=200 output=rdf_results.csv
# partial RDF:
defectmof-rdf trajectory=traj.traj elements=[Al,O]
```

### `defectmof-hierarchical` — Hierarchical optimization

```bash
defectmof-hierarchical --config-dir=configs --config-name=hierarchical
# or:
defectmof-hierarchical defective=defective.cif pristine=pristine.cif \
    target_size=[8,8,8] defect_fraction=0.44 backend=torchsim output=large.cif
```

---

## Hydra Config Reference

All keys are optional in the YAML file; unset keys fall back to the defaults listed here.

### Build config (`defectmof-build`)

```yaml
defective: path/to/defective.cif   # required
pristine:  path/to/pristine.cif    # required
size:      [4, 4, 4]               # supercell repeat
defect_fraction: 0.44              # fraction of defective cells
mode:      random                  # arrangement mode
seed:      42                      # random seed
output:    supercell.cif           # output CIF path
```

### Optimize config (`defectmof-optimize`)

```yaml
input:      supercell.cif          # required — input CIF
model:      mace_mp_medium         # mace_mp_small | mace_mp_medium | /path/to/model
optimizer:  lbfgs                  # lbfgs | fire | bfgs
fmax:       0.05                   # force convergence in eV/Å
max_steps:  500                    # maximum optimization steps
backend:    ase                    # ase | torchsim
device:     cuda                   # cuda | cpu
output:     optimized.cif          # output CIF path (optional)
```

### MD config (`defectmof-md`)

```yaml
input:              optimized.cif  # required — input CIF
temperature:        150.0          # temperature in Kelvin
n_steps:            5000           # production MD steps
timestep:           1.0            # timestep in femtoseconds
model:              mace_mp_medium
thermostat:         langevin       # langevin | nose_hoover
friction:           0.01           # Langevin friction (1/fs)
log_interval:       10             # snapshot every N steps
equilibration_steps: 1000          # discard first N steps
backend:            ase
device:             cuda
trajectory:         traj.traj      # trajectory output (optional)
```

### PDF config (`defectmof-pdf`)

```yaml
trajectory: traj.traj              # required — trajectory or list of CIFs
rmin:       1.0                    # minimum r in Å
rmax:       30.0                   # maximum r in Å
rstep:      0.01                   # r grid step in Å
scattering: xray                   # xray | neutron (diffpy engine only)
engine:     diffpy                 # diffpy | debye_calculator
device:     cuda                   # compute device for debye_calculator
output:     pdf_results.csv        # output CSV path
```

### RDF config (`defectmof-rdf`)

```yaml
trajectory: traj.traj              # required — trajectory or list of CIFs
rmax:       10.0                   # maximum r in Å
nbins:      200                    # number of histogram bins
elements:   null                   # e.g. [Al, O] for partial RDF
output:     rdf_results.csv        # output CSV path
```

### Hierarchical config (`defectmof-hierarchical`)

```yaml
defective:       path/to/defective.cif  # required
pristine:        path/to/pristine.cif   # required
target_size:     [8, 8, 8]              # final supercell (each dim >= 4, even)
defect_fraction: 0.44
mode:            random
model:           mace_mp_medium
backend:         ase
device:          cuda
output:          large_optimized.cif    # optional
```

---

## Examples

The `examples/structures/` directory contains four CIF files (see [Included Structures](#included-structures)). The original research script `run_4x4x4_new_2_percentages_correct.py` in the repository root shows the workflow that defectmof was built to replace: reading the two unit cells, building all ten arrangement modes across multiple seeds, pruning overlaps, fixing C-Br bonds, and running BFGS optimization with MACE.

### Example 1: Build all modes and compare

```python
import defectmof
from ase.io import write

defective = "examples/structures/struct1_optimized_c_axis.cif"
pristine  = "examples/structures/struct2_optimized_a_axis.cif"

modes = [
    "random", "alternating_a", "alternating_b", "alternating_c",
    "alternating_ab", "alternating_ac", "alternating_bc", "alternating_abc",
    "clustered_small", "clustered_large",
]

for mode in modes:
    sc = defectmof.build_supercell(
        defective, pristine,
        size=(4, 4, 4),
        defect_fraction=0.44,
        mode=mode,
        seed=42,
    )
    write(f"supercell_{mode}.cif", sc)
    print(f"{mode}: {len(sc)} atoms")
```

### Example 2: Full simulation pipeline

```python
import defectmof
import numpy as np
import matplotlib.pyplot as plt

# 1. Build supercell
sc = defectmof.build_supercell(
    "examples/structures/struct1_optimized_c_axis.cif",
    "examples/structures/struct2_optimized_a_axis.cif",
    size=(4, 4, 4), defect_fraction=0.44,
)

# 2. Optimize
optimized = defectmof.optimize(sc, model="mace_mp_medium", fmax=0.05,
                               device="cuda", output="optimized.cif")

# 3. Run MD at 150 K
snapshots = defectmof.run_md(
    optimized, temperature=150.0, n_steps=10000,
    equilibration_steps=2000, log_interval=20,
    trajectory="md_150K.traj",
)

# 4. Compute PDF
r, G_r = defectmof.compute_pdf(snapshots, rmin=1.0, rmax=30.0, engine="diffpy")
np.savetxt("pdf_150K.csv", np.column_stack([r, G_r]), header="r(A)  G(r)")

# 5. Plot
plt.plot(r, G_r)
plt.xlabel("r (Å)")
plt.ylabel("G(r)")
plt.title("PDF at 150 K")
plt.savefig("pdf_150K.png", dpi=150)
```

### Example 3: Large supercell via hierarchical optimization

```python
import defectmof

# Two-stage optimization: 2x2x2 then tile to 8x8x8
result = defectmof.hierarchical_optimize(
    "examples/structures/struct1_optimized_c_axis.cif",
    "examples/structures/struct2_optimized_a_axis.cif",
    target_size=(8, 8, 8),
    defect_fraction=0.44,
    mode="random",
    model="mace_mp_medium",
    backend="torchsim",  # use torchsim for memory-safe GPU batching
    device="cuda",
    output="large_8x8x8.cif",
)
print(f"Final structure: {len(result)} atoms")
```

### Example 4: Partial RDF for metal-oxygen coordination

```python
import defectmof
import matplotlib.pyplot as plt

r, g_r = defectmof.compute_rdf(
    "md_150K.traj",
    rmax=6.0,
    nbins=300,
    elements=("Al", "O"),
)

plt.plot(r, g_r)
plt.xlabel("r (Å)")
plt.ylabel("g(r) Al-O")
plt.title("Partial RDF: Al-O")
plt.savefig("rdf_AlO.png", dpi=150)
```

---

## Included Structures

The following CIF files are located in `examples/structures/`:

| File | Description |
|------|-------------|
| `struct1_optimized_c_axis.cif` | Defective MOF unit cell, optimized along the c-axis. Used as the `defective` input in the original research workflow |
| `struct2_optimized_a_axis.cif` | Pristine MOF unit cell, optimized along the a-axis. Used as the `pristine` input |
| `CONTCAR_final.cif` | Final VASP CONTCAR structure converted to CIF format |
| `CONTCAR_optimized.cif` | Intermediate optimized CONTCAR structure converted to CIF format |

The two primary structures (`struct1` and `struct2`) share compatible lattice parameters, which is required by `build_supercell`. They correspond to the `s1` and `s2` cells from the original `run_4x4x4_new_2_percentages_correct.py` workflow.

---

## Benchmark Reference

Performance measurements on an RTX 5090 (24 GB VRAM) for `mace_mp_medium`, `backend="torchsim"`:

| Task | Supercell | Atoms | VRAM | Wall time |
|------|-----------|-------|------|-----------|
| `optimize` (fmax=0.05) | 4×4×4 | ~4 000 | ~6 GB | ~8 min |
| `optimize` (fmax=0.05) | 8×8×8 | ~32 000 | ~20 GB | ~90 min |
| `hierarchical_optimize` | → 8×8×8 | ~32 000 | ~6 GB peak | ~25 min |
| `run_md` (5 000 steps, 150 K) | 4×4×4 | ~4 000 | ~6 GB | ~12 min |
| `compute_pdf` (diffpy, 100 frames) | — | ~4 000 | CPU only | ~3 min |
| `compute_pdf` (debye_calculator, 100 frames) | — | ~4 000 | ~2 GB | ~30 s |

Notes:
- `hierarchical_optimize` reduces peak VRAM by optimizing a small cell first and then tiling.
- For systems that exceed GPU memory, set `memory_limit_gb` to cap memory use, or use `device="cpu"`.
- Benchmarks are indicative. Actual performance depends on convergence and structure complexity.

---

## Contributing

### Workflow

1. Fork the repository and create a feature branch: `git checkout -b feat/my-feature`
2. Make changes and add tests in `tests/`.
3. Run the test suite: `pytest tests/ -v`
4. Lint your code: `ruff check src/ tests/`
5. Open a pull request against `master`.

### Branch protection

The `master` branch is protected:
- All PRs require at least one approving review before merging.
- The CI workflow (Python 3.11, 3.12, 3.13) must pass.
- Direct pushes to `master` are not permitted.

### Adding a new arrangement mode

1. Add the mode name to the `valid_modes` set in `supercell.py`.
2. Add the pattern logic in the `_generate_arrangements` loop.
3. Add a test in `tests/test_supercell.py`.
4. Document the mode in this README under the `build_supercell` API table.

### Adding a new PDF engine

1. Add a `_pdf_<engine>` function in `pdf.py`.
2. Dispatch to it in `compute_pdf`.
3. Add the optional dependency to `pyproject.toml` under `[project.optional-dependencies]`.
4. Add tests in `tests/test_pdf.py`.

---

## License

MIT
