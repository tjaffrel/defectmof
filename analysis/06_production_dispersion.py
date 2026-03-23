"""Production analysis with consistent dispersion corrections.

Dispersion strategy (no double counting):
- mace_mp_medium: PBE+U → add dispersion=True (D3(BJ))
- mofs_v2 / pbe_d3: PBE+D3(BJ) baked in → no external D3
- mof_omat_v2 / pbe_d3: PBE+D3(BJ) baked in → no external D3

All runs: L-BFGS, fmax=0.05, max_steps=10000
MD: 150K, 10 ps production (10000 steps * 1 fs), 1 ps equilibration
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
from pathlib import Path
from collections import Counter

from defectmof import optimize, run_md, compute_rdf, compute_pdf

STRUCTURES = "examples/structures"
MODELS = {
    "mace_mp_medium_D3": {
        "path": "mace_mp_medium", "head": None,
        "dispersion": True,  # Add D3(BJ) externally
        "theory": "PBE+U+D3(BJ)",
    },
    "mofs_v2_pbe_d3": {
        "path": "models/mofs_v2.model", "head": "pbe_d3",
        "dispersion": False,  # D3 already in the head
        "theory": "PBE+D3(BJ) [fine-tuned]",
    },
    "mof_omat_v2_pbe_d3": {
        "path": "models/mof-omat-0-v2.model", "head": "pbe_d3",
        "dispersion": False,  # D3 already in the head
        "theory": "PBE+D3(BJ) [fine-tuned on OMAT]",
    },
}
STRUCTURES_MAP = {
    "pristine": f"{STRUCTURES}/CONTCAR_optimized.cif",
    "defective": f"{STRUCTURES}/CONTCAR_final.cif",
}

os.makedirs("analysis/production", exist_ok=True)

VRAM_LIMIT_GB = 20.0


def gpu_status():
    used = torch.cuda.memory_allocated(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {used:.1f}/{total:.1f} GB")
    return used


# ===== PART 1: Optimization (L-BFGS, fmax=0.05, 10000 steps) =====
print("=" * 70)
print("PART 1: STRUCTURE OPTIMIZATION")
print("  L-BFGS, fmax=0.05, max_steps=10000")
print("=" * 70)

opt_results = {}

for model_name, cfg in MODELS.items():
    for struct_name, cif_path in STRUCTURES_MAP.items():
        torch.cuda.empty_cache()

        print(f"\n--- {struct_name} / {model_name} ({cfg['theory']}) ---")
        gpu_status()

        t0 = time.time()
        try:
            result = optimize(
                atoms=cif_path,
                model=cfg["path"],
                optimizer="lbfgs",
                fmax=0.05,
                max_steps=10000,
                device="cuda",
                head=cfg["head"],
                dispersion=cfg["dispersion"],
                output=f"analysis/production/{struct_name}_{model_name}_optimized.cif",
            )
            dt = time.time() - t0

            # Check convergence
            from defectmof._ase import _get_mace_calculator
            result.calc = _get_mace_calculator(
                cfg["path"], "cuda", head=cfg["head"], dispersion=cfg["dispersion"],
            )
            energy = result.get_potential_energy()
            fmax = float(np.abs(result.get_forces()).max())
            e_per_atom = energy / len(result)

            opt_results[(model_name, struct_name)] = {
                "time": dt, "fmax": fmax, "energy": energy,
                "e_per_atom": e_per_atom, "n_atoms": len(result),
            }
            print(f"  Done: {dt:.1f}s | fmax={fmax:.4f} eV/A | E/atom={e_per_atom:.3f} eV")

        except torch.cuda.OutOfMemoryError:
            print(f"  OUT OF MEMORY — skipping")
            opt_results[(model_name, struct_name)] = None
        except Exception as e:
            print(f"  FAILED: {e}")
            opt_results[(model_name, struct_name)] = None

    torch.cuda.empty_cache()

# Print optimization summary table
print(f"\n{'=' * 70}")
print("OPTIMIZATION SUMMARY")
print(f"{'=' * 70}")
print(f"{'Model':<28} {'Structure':<12} {'Time':>8} {'fmax':>10} {'E/atom':>10}")
print("-" * 70)
for (m, s), r in opt_results.items():
    if r is None:
        print(f"{m:<28} {s:<12} {'FAILED':>8}")
    else:
        print(f"{m:<28} {s:<12} {r['time']:>7.1f}s {r['fmax']:>10.4f} {r['e_per_atom']:>10.3f}")


# ===== PART 2: MD at 150K (10 ps production) =====
print(f"\n{'=' * 70}")
print("PART 2: MD SIMULATION AT 150K")
print("  10 ps production, 1 ps equilibration, Langevin thermostat")
print("=" * 70)

all_snapshots = {}

for model_name, cfg in MODELS.items():
    for struct_name, cif_path in STRUCTURES_MAP.items():
        torch.cuda.empty_cache()
        key = (model_name, struct_name)

        # Use optimized structure if available
        opt_cif = f"analysis/production/{struct_name}_{model_name}_optimized.cif"
        input_path = opt_cif if os.path.exists(opt_cif) else cif_path

        print(f"\n--- MD {struct_name} / {model_name} ({cfg['theory']}) ---")
        gpu_status()

        t0 = time.time()
        try:
            snapshots = run_md(
                atoms=input_path,
                temperature=150.0,
                n_steps=10000,           # 10 ps production
                timestep=1.0,            # 1 fs
                model=cfg["path"],
                thermostat="langevin",
                friction=0.01,
                log_interval=50,         # save every 50 fs → 200 snapshots
                equilibration_steps=1000, # 1 ps equilibration
                device="cuda",
                head=cfg["head"],
                dispersion=cfg["dispersion"],
            )
            dt = time.time() - t0
            all_snapshots[key] = snapshots
            print(f"  {len(snapshots)} snapshots in {dt:.1f}s")

        except torch.cuda.OutOfMemoryError:
            print(f"  OUT OF MEMORY — skipping")
            all_snapshots[key] = None
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            all_snapshots[key] = None

    torch.cuda.empty_cache()


# ===== PART 3: RDF + PDF =====
print(f"\n{'=' * 70}")
print("PART 3: COMPUTING RDF AND PDF")
print("=" * 70)

all_rdf = {}
all_pdf = {}
colors = {"mace_mp_medium_D3": "blue", "mofs_v2_pbe_d3": "green", "mof_omat_v2_pbe_d3": "orange"}

for key, snapshots in all_snapshots.items():
    if snapshots is None:
        continue
    model_name, struct_name = key

    # RDF
    r, g_r = compute_rdf(snapshots, rmax=10.0, nbins=500)
    all_rdf[key] = (r, g_r)
    np.savetxt(f"analysis/production/rdf_{struct_name}_{model_name}.csv",
               np.column_stack([r, g_r]), header="r(A)  g(r)", delimiter="  ")

    # PDF
    try:
        r_pdf, g_pdf = compute_pdf(
            snapshots, rmin=1.0, rmax=20.0, rstep=0.05,
            scattering="xray", engine="diffpy",
        )
        all_pdf[key] = (r_pdf, g_pdf)
        np.savetxt(f"analysis/production/pdf_{struct_name}_{model_name}.csv",
                   np.column_stack([r_pdf, g_pdf]), header="r(A)  G(r)", delimiter="  ")
    except Exception as e:
        print(f"  PDF failed for {key}: {e}")

    print(f"  {struct_name}/{model_name}: RDF + PDF done")


# ===== PART 4: Comparison plots =====
print(f"\n{'=' * 70}")
print("PART 4: GENERATING PLOTS")
print("=" * 70)

# 4a. Per-model: pristine vs defective (RDF + diff)
for model_name in MODELS:
    key_p = (model_name, "pristine")
    key_d = (model_name, "defective")
    if key_p not in all_rdf or key_d not in all_rdf:
        continue

    theory = MODELS[model_name]["theory"]
    r_p, g_p = all_rdf[key_p]
    r_d, g_d = all_rdf[key_d]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(r_p, g_p, "b-", lw=1.5, label="Pristine")
    axes[0].plot(r_d, g_d, "r-", lw=1.5, label="Defective")
    axes[0].set_ylabel("g(r)", fontsize=12)
    axes[0].set_title(f"RDF — {model_name} [{theory}] — 150K, 10ps", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].set_xlim(0, 10)

    r_c = np.linspace(0, 10, 500)
    diff = np.interp(r_c, r_d, g_d) - np.interp(r_c, r_p, g_p)
    axes[1].fill_between(r_c, diff, 0, where=diff > 0, alpha=0.3, color="red", label="Defective > Pristine")
    axes[1].fill_between(r_c, diff, 0, where=diff < 0, alpha=0.3, color="blue", label="Pristine > Defective")
    axes[1].plot(r_c, diff, "k-", lw=0.8)
    axes[1].axhline(0, color="gray", ls="--", lw=0.5)
    axes[1].set_xlabel("r (Å)", fontsize=12)
    axes[1].set_ylabel("Δg(r)", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, 10)
    plt.tight_layout()
    plt.savefig(f"analysis/production/rdf_comparison_{model_name}.png", dpi=150)
    plt.close()

# 4b. Per-model: pristine vs defective PDF
for model_name in MODELS:
    key_p = (model_name, "pristine")
    key_d = (model_name, "defective")
    if key_p not in all_pdf or key_d not in all_pdf:
        continue

    theory = MODELS[model_name]["theory"]
    r_p, g_p = all_pdf[key_p]
    r_d, g_d = all_pdf[key_d]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(r_p, g_p, "b-", lw=1.2, label="Pristine")
    axes[0].plot(r_d, g_d, "r-", lw=1.2, label="Defective")
    axes[0].axhline(0, color="gray", ls="--", lw=0.5)
    axes[0].set_ylabel("G(r)", fontsize=12)
    axes[0].set_title(f"PDF (X-ray) — {model_name} [{theory}] — 150K, 10ps", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].set_xlim(1, 20)

    r_c = np.linspace(1, 20, 400)
    diff = np.interp(r_c, r_d, g_d) - np.interp(r_c, r_p, g_p)
    axes[1].fill_between(r_c, diff, 0, where=diff > 0, alpha=0.3, color="red")
    axes[1].fill_between(r_c, diff, 0, where=diff < 0, alpha=0.3, color="blue")
    axes[1].plot(r_c, diff, "k-", lw=0.8)
    axes[1].axhline(0, color="gray", ls="--", lw=0.5)
    axes[1].set_xlabel("r (Å)", fontsize=12)
    axes[1].set_ylabel("ΔG(r)", fontsize=12)
    axes[1].set_xlim(1, 20)
    plt.tight_layout()
    plt.savefig(f"analysis/production/pdf_comparison_{model_name}.png", dpi=150)
    plt.close()

# 4c. Model comparison: RDF + PDF side by side
for struct_name in STRUCTURES_MAP:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for model_name in MODELS:
        key = (model_name, struct_name)
        label = f"{model_name}\n[{MODELS[model_name]['theory']}]"
        if key in all_rdf:
            r, g = all_rdf[key]
            axes[0].plot(r, g, color=colors[model_name], lw=1.2, label=label)
        if key in all_pdf:
            r, g = all_pdf[key]
            axes[1].plot(r, g, color=colors[model_name], lw=1.2, label=label)

    axes[0].set_xlabel("r (Å)", fontsize=12)
    axes[0].set_ylabel("g(r)", fontsize=12)
    axes[0].set_title(f"RDF — {struct_name} — Model Comparison (all with D3)", fontsize=12)
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(0, 10)

    axes[1].set_xlabel("r (Å)", fontsize=12)
    axes[1].set_ylabel("G(r)", fontsize=12)
    axes[1].set_title(f"PDF (X-ray) — {struct_name} — Model Comparison (all with D3)", fontsize=12)
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(1, 20)
    axes[1].axhline(0, color="gray", ls="--", lw=0.5)

    plt.tight_layout()
    plt.savefig(f"analysis/production/model_comparison_{struct_name}.png", dpi=150)
    plt.close()


# ===== FINAL SUMMARY =====
print(f"\n{'=' * 70}")
print("PRODUCTION ANALYSIS COMPLETE")
print("=" * 70)
print("\nDispersion corrections:")
for m, cfg in MODELS.items():
    print(f"  {m}: {cfg['theory']}")
print("\nOutput files:")
for f in sorted(Path("analysis/production").glob("*")):
    print(f"  {f.name}")
