"""MD simulation at 150K for defective and pristine MOF-303, then compute PDFs.

Answers the collaborator's question:
"Try the MD simulation approach at 150K for these 2 structures so that
we may then calculate the PDFs based on the MD snapshots."

Uses MACE-MP0 small model for testing on RTX 5090.
"""

from defectmof import run_md, compute_rdf
import matplotlib.pyplot as plt
import numpy as np
import time

STRUCTURES = "examples/structures"
MODEL = "mace_mp_small"  # Use small for local testing, medium for production

# ===== 1. MD simulation at 150K =====

for name, cif in [
    ("pristine", f"{STRUCTURES}/CONTCAR_optimized.cif"),
    ("defective", f"{STRUCTURES}/CONTCAR_final.cif"),
]:
    print(f"\n{'='*60}")
    print(f"Running MD at 150K for {name} MOF-303")
    print(f"{'='*60}")

    t0 = time.time()
    snapshots = run_md(
        atoms=cif,
        temperature=150.0,
        n_steps=2000,            # 2 ps production (2000 steps * 1 fs)
        timestep=1.0,            # 1 fs
        model=MODEL,
        thermostat="langevin",
        friction=0.01,
        log_interval=10,         # save every 10 steps = 200 snapshots
        equilibration_steps=500, # 0.5 ps equilibration
        backend="ase",
        device="cuda",
        trajectory=f"analysis/md_150K_{name}.traj",
    )
    dt = time.time() - t0
    print(f"  Collected {len(snapshots)} snapshots in {dt:.1f}s")
    print(f"  Trajectory saved to analysis/md_150K_{name}.traj")

    # ===== 2. Compute RDF from MD snapshots =====
    print(f"\nComputing RDF for {name}...")

    # Total RDF
    r, g_r = compute_rdf(snapshots, rmax=10.0, nbins=500)
    np.savetxt(f"analysis/rdf_150K_{name}.csv",
               np.column_stack([r, g_r]),
               header="r(A)  g(r)", delimiter="  ")

    # Partial RDFs for key pairs
    partials = {}
    for pair in [("Al", "O"), ("Al", "N"), ("C", "O"), ("C", "Br")]:
        try:
            r_p, g_p = compute_rdf(snapshots, rmax=8.0, nbins=400, elements=pair)
            partials[f"{pair[0]}-{pair[1]}"] = (r_p, g_p)
            np.savetxt(f"analysis/rdf_150K_{name}_{pair[0]}_{pair[1]}.csv",
                       np.column_stack([r_p, g_p]),
                       header=f"r(A)  g_{pair[0]}-{pair[1]}(r)", delimiter="  ")
        except Exception as e:
            print(f"  Skipping {pair}: {e}")

    # ===== 3. Plot =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total RDF
    axes[0].plot(r, g_r, 'k-', linewidth=1.5)
    axes[0].set_xlabel("r (Å)")
    axes[0].set_ylabel("g(r)")
    axes[0].set_title(f"Total RDF — {name} MOF-303 at 150K")
    axes[0].set_xlim(0, 10)

    # Partial RDFs
    colors = {"Al-O": "blue", "Al-N": "green", "C-O": "red", "C-Br": "orange"}
    for label, (r_p, g_p) in partials.items():
        axes[1].plot(r_p, g_p, label=label, color=colors.get(label, None), linewidth=1.2)
    axes[1].set_xlabel("r (Å)")
    axes[1].set_ylabel("g(r)")
    axes[1].set_title(f"Partial RDFs — {name} MOF-303 at 150K")
    axes[1].legend()
    axes[1].set_xlim(0, 8)

    plt.tight_layout()
    plt.savefig(f"analysis/rdf_150K_{name}.png", dpi=150)
    plt.close()
    print(f"  Plots saved to analysis/rdf_150K_{name}.png")

print("\n" + "="*60)
print("DONE — Compare pristine vs defective PDFs to see defect signatures")
print("="*60)
