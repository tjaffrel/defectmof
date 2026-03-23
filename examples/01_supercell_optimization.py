"""Example: Build and optimize a 2x2x2 MOF supercell with defects.

This builds a supercell with 44% defective / 56% pristine unit cells
using a random arrangement, then optimizes with MACE-MP0.
"""
from defectmof import build_supercell, optimize

# Build a 2x2x2 supercell with random defect placement
supercell = build_supercell(
    defective="examples/structures/struct1_optimized_c_axis.cif",
    pristine="examples/structures/struct2_optimized_a_axis.cif",
    size=(2, 2, 2),
    defect_fraction=0.44,
    mode="random",
)
print(f"Built supercell: {len(supercell)} atoms")

# Optimize with MACE (use 'small' model for testing)
optimized = optimize(
    supercell,
    model="mace_mp_small",
    optimizer="lbfgs",
    fmax=0.05,
    device="cuda",
    output="supercell_2x2x2_optimized.cif",
)
print(f"Optimized: {len(optimized)} atoms")
