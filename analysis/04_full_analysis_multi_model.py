"""Full analysis: MD at 150K with 3 MACE models, RDF + PDF (diffpy), combined plots.

Models tested:
  1. mace_mp_small (MACE-MP0 small — baseline)
  2. models/mofs_v2.model (MOF-specific fine-tuned)
  3. models/mof-omat-0-v2.model (MOF+OMAT fine-tuned)

Also tests the Hydra YAML config workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from omegaconf import OmegaConf

from defectmof import run_md, compute_rdf, compute_pdf

STRUCTURES = "examples/structures"
MODELS = {
    "mace_mp_small": {"path": "mace_mp_small", "head": None},
    "mofs_v2": {"path": "models/mofs_v2.model", "head": "pt_head"},
    "mof_omat_v2": {"path": "models/mof-omat-0-v2.model", "head": "pt_head"},
}
STRUCTURES_MAP = {
    "pristine": f"{STRUCTURES}/CONTCAR_optimized.cif",
    "defective": f"{STRUCTURES}/CONTCAR_final.cif",
}

os.makedirs("analysis/results", exist_ok=True)

# ===== 1. Test Hydra YAML config loading =====
print("=" * 60)
print("Testing Hydra YAML config loading")
print("=" * 60)

for cfg_name in ["md_pristine_150K", "md_defective_150K"]:
    cfg = OmegaConf.load(f"analysis/configs/{cfg_name}.yaml")
    print(f"  {cfg_name}: input={cfg.input}, T={cfg.temperature}K, model={cfg.model}")

# ===== 2. Run MD at 150K for each model x structure =====
all_snapshots = {}  # {(model_name, struct_name): [Atoms, ...]}

for model_name, model_cfg in MODELS.items():
    for struct_name, cif_path in STRUCTURES_MAP.items():
        key = (model_name, struct_name)
        print(f"\n{'=' * 60}")
        print(f"MD 150K: {struct_name} with {model_name}")
        print(f"{'=' * 60}")

        t0 = time.time()
        try:
            snapshots = run_md(
                atoms=cif_path,
                temperature=150.0,
                n_steps=2000,
                timestep=1.0,
                model=model_cfg["path"],
                thermostat="langevin",
                friction=0.01,
                log_interval=10,
                equilibration_steps=500,
                backend="ase",
                device="cuda",
                head=model_cfg["head"],
            )
            dt = time.time() - t0
            all_snapshots[key] = snapshots
            print(f"  {len(snapshots)} snapshots in {dt:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            all_snapshots[key] = None

# ===== 3. Compute RDF for all =====
print(f"\n{'=' * 60}")
print("Computing RDFs...")
print("=" * 60)

all_rdf = {}
for key, snapshots in all_snapshots.items():
    if snapshots is None:
        continue
    model_name, struct_name = key
    r, g_r = compute_rdf(snapshots, rmax=10.0, nbins=500)
    all_rdf[key] = (r, g_r)
    np.savetxt(
        f"analysis/results/rdf_{struct_name}_{model_name}.csv",
        np.column_stack([r, g_r]),
        header="r(A)  g(r)", delimiter="  ",
    )
    print(f"  {struct_name}/{model_name}: done")

# ===== 4. Compute PDF (diffpy) for all =====
print(f"\n{'=' * 60}")
print("Computing PDFs (diffpy X-ray)...")
print("=" * 60)

all_pdf = {}
for key, snapshots in all_snapshots.items():
    if snapshots is None:
        continue
    model_name, struct_name = key
    try:
        r_pdf, g_pdf = compute_pdf(
            snapshots, rmin=1.0, rmax=30.0, rstep=0.05,
            scattering="xray", engine="diffpy",
        )
        all_pdf[key] = (r_pdf, g_pdf)
        np.savetxt(
            f"analysis/results/pdf_{struct_name}_{model_name}.csv",
            np.column_stack([r_pdf, g_pdf]),
            header="r(A)  G(r)", delimiter="  ",
        )
        print(f"  {struct_name}/{model_name}: done")
    except Exception as e:
        print(f"  {struct_name}/{model_name} FAILED: {e}")

# ===== 5. Combined RDF comparison plots =====

# 5a. Pristine vs Defective for each model
for model_name in MODELS:
    key_p = (model_name, "pristine")
    key_d = (model_name, "defective")
    if key_p not in all_rdf or key_d not in all_rdf:
        continue

    r_p, g_p = all_rdf[key_p]
    r_d, g_d = all_rdf[key_d]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(r_p, g_p, "b-", linewidth=1.5, label="Pristine")
    axes[0].plot(r_d, g_d, "r-", linewidth=1.5, label="Defective")
    axes[0].set_ylabel("g(r)", fontsize=12)
    axes[0].set_title(f"RDF — {model_name} — 150K", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].set_xlim(0, 10)

    r_common = np.linspace(0, 10, 500)
    g_p_i = np.interp(r_common, r_p, g_p)
    g_d_i = np.interp(r_common, r_d, g_d)
    diff = g_d_i - g_p_i

    axes[1].fill_between(r_common, diff, 0, where=diff > 0, alpha=0.3, color="red", label="Defective > Pristine")
    axes[1].fill_between(r_common, diff, 0, where=diff < 0, alpha=0.3, color="blue", label="Pristine > Defective")
    axes[1].plot(r_common, diff, "k-", linewidth=0.8)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.5)
    axes[1].set_xlabel("r (Å)", fontsize=12)
    axes[1].set_ylabel("Δg(r)", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig(f"analysis/results/rdf_comparison_{model_name}.png", dpi=150)
    plt.close()

# 5b. Model comparison for each structure
for struct_name in STRUCTURES_MAP:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RDF comparison
    colors = {"mace_mp_small": "blue", "mofs_v2": "green", "mof_omat_v2": "orange"}
    for model_name in MODELS:
        key = (model_name, struct_name)
        if key not in all_rdf:
            continue
        r, g_r = all_rdf[key]
        axes[0].plot(r, g_r, color=colors[model_name], linewidth=1.2, label=model_name)

    axes[0].set_xlabel("r (Å)", fontsize=12)
    axes[0].set_ylabel("g(r)", fontsize=12)
    axes[0].set_title(f"RDF — {struct_name} — Model Comparison", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].set_xlim(0, 10)

    # PDF comparison
    for model_name in MODELS:
        key = (model_name, struct_name)
        if key not in all_pdf:
            continue
        r, g_r = all_pdf[key]
        axes[1].plot(r, g_r, color=colors[model_name], linewidth=1.2, label=model_name)

    axes[1].set_xlabel("r (Å)", fontsize=12)
    axes[1].set_ylabel("G(r)", fontsize=12)
    axes[1].set_title(f"PDF (X-ray, diffpy) — {struct_name} — Model Comparison", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].set_xlim(1, 30)

    plt.tight_layout()
    plt.savefig(f"analysis/results/model_comparison_{struct_name}.png", dpi=150)
    plt.close()

# 5c. PDF pristine vs defective overlay for each model
for model_name in MODELS:
    key_p = (model_name, "pristine")
    key_d = (model_name, "defective")
    if key_p not in all_pdf or key_d not in all_pdf:
        continue

    r_p, g_p = all_pdf[key_p]
    r_d, g_d = all_pdf[key_d]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(r_p, g_p, "b-", linewidth=1.2, label="Pristine")
    axes[0].plot(r_d, g_d, "r-", linewidth=1.2, label="Defective")
    axes[0].set_ylabel("G(r)", fontsize=12)
    axes[0].set_title(f"PDF (X-ray) — {model_name} — 150K", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].set_xlim(1, 30)

    # Difference
    r_common = np.linspace(1, 30, 600)
    g_p_i = np.interp(r_common, r_p, g_p)
    g_d_i = np.interp(r_common, r_d, g_d)
    diff = g_d_i - g_p_i

    axes[1].fill_between(r_common, diff, 0, where=diff > 0, alpha=0.3, color="red")
    axes[1].fill_between(r_common, diff, 0, where=diff < 0, alpha=0.3, color="blue")
    axes[1].plot(r_common, diff, "k-", linewidth=0.8)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.5)
    axes[1].set_xlabel("r (Å)", fontsize=12)
    axes[1].set_ylabel("ΔG(r)", fontsize=12)
    axes[1].set_title("Difference: G_defective(r) - G_pristine(r)", fontsize=11)
    axes[1].set_xlim(1, 30)

    plt.tight_layout()
    plt.savefig(f"analysis/results/pdf_comparison_{model_name}.png", dpi=150)
    plt.close()

# ===== 6. Summary =====
print(f"\n{'=' * 60}")
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\nOutput files in analysis/results/:")
for f in sorted(Path("analysis/results").glob("*")):
    print(f"  {f.name}")
