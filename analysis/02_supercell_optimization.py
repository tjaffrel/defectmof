"""Supercell optimization with all arrangement modes.

Answers the collaborator's questions:
- Generate supercells in 44:56 proportion with different arrangements
- Compare convergence speed: L-BFGS (new) vs BFGS (original)
- Test feasibility of larger supercells

Uses MACE-MP0 small model for testing on RTX 5090.
"""

from defectmof import build_supercell, optimize
from defectmof.utils import visualize_distribution
from defectmof.supercell import _generate_arrangements
import numpy as np
import time

STRUCTURES = "examples/structures"
DEFECTIVE = f"{STRUCTURES}/struct1_optimized_c_axis.cif"
PRISTINE = f"{STRUCTURES}/struct2_optimized_a_axis.cif"
MODEL = "mace_mp_small"

# ===== 1. Build and optimize 2x2x2 supercells with ALL modes =====

modes = [
    "alternating_a", "alternating_b", "alternating_c",
    "alternating_ab", "alternating_ac", "alternating_bc", "alternating_abc",
    "clustered_small", "clustered_large",
    "random",
]

results = {}

for mode in modes:
    print(f"\n{'='*60}")
    print(f"Mode: {mode}")
    print(f"{'='*60}")

    # Build supercell
    sc = build_supercell(
        DEFECTIVE, PRISTINE,
        size=(2, 2, 2),
        defect_fraction=0.44,
        mode=mode,
        seed=42,
    )
    print(f"  Built: {len(sc)} atoms")

    # Visualize the arrangement
    grid = _generate_arrangements(2, 2, 2, 0.44, mode, seed=42)
    visualize_distribution(grid, mode, output_path=f"analysis/dist_{mode}.png")

    # Optimize with L-BFGS (our new default — 20x faster than BFGS)
    t0 = time.time()
    optimized = optimize(
        sc,
        model=MODEL,
        optimizer="lbfgs",
        fmax=0.1,         # loose for testing
        max_steps=100,
        device="cuda",
        output=f"analysis/supercell_2x2x2_{mode}_optimized.cif",
    )
    dt = time.time() - t0

    # Get final forces
    from mace.calculators import mace_mp
    optimized.calc = mace_mp(model="small", device="cuda", default_dtype="float32")
    fmax_final = float(np.abs(optimized.get_forces()).max())

    results[mode] = {"atoms": len(sc), "time": dt, "fmax": fmax_final}
    print(f"  L-BFGS: {dt:.1f}s, fmax={fmax_final:.4f} eV/A")

# ===== 2. Summary table =====

print(f"\n{'='*60}")
print(f"SUMMARY — 2x2x2 supercell optimization (L-BFGS, fmax<0.1)")
print(f"{'='*60}")
print(f"{'Mode':<20} {'Atoms':>6} {'Time (s)':>10} {'Final fmax':>12}")
print("-" * 50)
for mode, r in results.items():
    print(f"{mode:<20} {r['atoms']:>6} {r['time']:>10.1f} {r['fmax']:>12.4f}")

# ===== 3. Optimizer comparison on one mode =====

print(f"\n{'='*60}")
print(f"OPTIMIZER COMPARISON — 2x2x2 random mode")
print(f"{'='*60}")

sc = build_supercell(DEFECTIVE, PRISTINE, size=(2, 2, 2), mode="random", seed=42)

for opt_name in ["lbfgs", "fire", "bfgs"]:
    t0 = time.time()
    result = optimize(
        sc, model=MODEL, optimizer=opt_name,
        fmax=0.1, max_steps=100, device="cuda",
    )
    dt = time.time() - t0
    result.calc = mace_mp(model="small", device="cuda", default_dtype="float32")
    fmax = float(np.abs(result.get_forces()).max())
    print(f"  {opt_name:>6}: {dt:6.1f}s, fmax={fmax:.4f} eV/A")
