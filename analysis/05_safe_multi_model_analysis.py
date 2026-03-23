"""Safe multi-model analysis with GPU memory monitoring.

Models:
  1. mace_mp_medium (MACE-MP0 medium — production quality)
  2. models/mofs_v2.model (MOF fine-tuned, head=pt_head)
  3. models/mof-omat-0-v2.model (MOF+OMAT fine-tuned, head=pt_head)

Memory safety: checks GPU VRAM before each run, skips if > 20 GB used.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
from pathlib import Path

from defectmof import run_md, compute_rdf, compute_pdf

STRUCTURES = "examples/structures"
MODELS = {
    "mace_mp_medium": {"path": "mace_mp_medium", "head": None},
    "mofs_v2": {"path": "models/mofs_v2.model", "head": "pt_head"},
    "mof_omat_v2": {"path": "models/mof-omat-0-v2.model", "head": "pt_head"},
}
STRUCTURES_MAP = {
    "pristine": f"{STRUCTURES}/CONTCAR_optimized.cif",
    "defective": f"{STRUCTURES}/CONTCAR_final.cif",
}
VRAM_LIMIT_GB = 20.0  # Stop if VRAM usage exceeds this

os.makedirs("analysis/results", exist_ok=True)


def check_gpu_memory():
    """Return current GPU memory usage in GB. Abort if too high."""
    if not torch.cuda.is_available():
        return 0.0
    used = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU VRAM: {used:.1f} GB used, {reserved:.1f} GB reserved, {total:.1f} GB total")
    return reserved


def clear_gpu():
    """Force clear GPU cache."""
    torch.cuda.empty_cache()
    import gc
    gc.collect()


# ===== Run MD for each model x structure =====
all_snapshots = {}

for model_name, model_cfg in MODELS.items():
    for struct_name, cif_path in STRUCTURES_MAP.items():
        key = (model_name, struct_name)

        # Check memory before starting
        clear_gpu()
        vram = check_gpu_memory()
        if vram > VRAM_LIMIT_GB:
            print(f"  SKIPPING {model_name}/{struct_name}: VRAM too high ({vram:.1f} GB)")
            all_snapshots[key] = None
            continue

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
        except torch.cuda.OutOfMemoryError:
            print(f"  OUT OF MEMORY — skipping {model_name}/{struct_name}")
            clear_gpu()
            all_snapshots[key] = None
        except Exception as e:
            print(f"  FAILED: {e}")
            all_snapshots[key] = None

        # Check memory after
        check_gpu_memory()

    # Clear GPU between models
    clear_gpu()

# ===== Compute RDF and PDF =====
print(f"\n{'=' * 60}")
print("Computing RDFs and PDFs...")
print("=" * 60)

all_rdf = {}
all_pdf = {}

for key, snapshots in all_snapshots.items():
    if snapshots is None:
        continue
    model_name, struct_name = key

    # RDF
    r, g_r = compute_rdf(snapshots, rmax=10.0, nbins=500)
    all_rdf[key] = (r, g_r)
    np.savetxt(f"analysis/results/rdf_{struct_name}_{model_name}.csv",
               np.column_stack([r, g_r]), header="r(A)  g(r)", delimiter="  ")

    # PDF (diffpy X-ray)
    try:
        r_pdf, g_pdf = compute_pdf(
            snapshots, rmin=1.0, rmax=20.0, rstep=0.05,
            scattering="xray", engine="diffpy",
        )
        all_pdf[key] = (r_pdf, g_pdf)
        np.savetxt(f"analysis/results/pdf_{struct_name}_{model_name}.csv",
                   np.column_stack([r_pdf, g_pdf]), header="r(A)  G(r)", delimiter="  ")
        print(f"  {struct_name}/{model_name}: RDF + PDF done")
    except Exception as e:
        print(f"  {struct_name}/{model_name}: RDF done, PDF FAILED: {e}")

# ===== Plots =====

colors = {"mace_mp_medium": "blue", "mofs_v2": "green", "mof_omat_v2": "orange"}

# 1. Per-model: pristine vs defective (RDF + difference)
for model_name in MODELS:
    key_p = (model_name, "pristine")
    key_d = (model_name, "defective")
    if key_p not in all_rdf or key_d not in all_rdf:
        continue

    r_p, g_p = all_rdf[key_p]
    r_d, g_d = all_rdf[key_d]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(r_p, g_p, "b-", lw=1.5, label="Pristine")
    axes[0].plot(r_d, g_d, "r-", lw=1.5, label="Defective")
    axes[0].set_ylabel("g(r)", fontsize=12)
    axes[0].set_title(f"RDF — {model_name} — 150K", fontsize=13)
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
    plt.savefig(f"analysis/results/rdf_comparison_{model_name}.png", dpi=150)
    plt.close()

# 2. Per-model: pristine vs defective PDF
for model_name in MODELS:
    key_p = (model_name, "pristine")
    key_d = (model_name, "defective")
    if key_p not in all_pdf or key_d not in all_pdf:
        continue

    r_p, g_p = all_pdf[key_p]
    r_d, g_d = all_pdf[key_d]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(r_p, g_p, "b-", lw=1.2, label="Pristine")
    axes[0].plot(r_d, g_d, "r-", lw=1.2, label="Defective")
    axes[0].axhline(0, color="gray", ls="--", lw=0.5)
    axes[0].set_ylabel("G(r)", fontsize=12)
    axes[0].set_title(f"PDF (X-ray, diffpy) — {model_name} — 150K", fontsize=13)
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
    plt.savefig(f"analysis/results/pdf_comparison_{model_name}.png", dpi=150)
    plt.close()

# 3. Model comparison: RDF + PDF side by side for each structure
for struct_name in STRUCTURES_MAP:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for model_name in MODELS:
        key = (model_name, struct_name)
        if key in all_rdf:
            r, g = all_rdf[key]
            axes[0].plot(r, g, color=colors[model_name], lw=1.2, label=model_name)
        if key in all_pdf:
            r, g = all_pdf[key]
            axes[1].plot(r, g, color=colors[model_name], lw=1.2, label=model_name)

    axes[0].set_xlabel("r (Å)", fontsize=12)
    axes[0].set_ylabel("g(r)", fontsize=12)
    axes[0].set_title(f"RDF — {struct_name} — Model Comparison", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].set_xlim(0, 10)

    axes[1].set_xlabel("r (Å)", fontsize=12)
    axes[1].set_ylabel("G(r)", fontsize=12)
    axes[1].set_title(f"PDF (X-ray) — {struct_name} — Model Comparison", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].set_xlim(1, 20)
    axes[1].axhline(0, color="gray", ls="--", lw=0.5)

    plt.tight_layout()
    plt.savefig(f"analysis/results/model_comparison_{struct_name}.png", dpi=150)
    plt.close()

print(f"\n{'=' * 60}")
print("ANALYSIS COMPLETE")
print("=" * 60)
for f in sorted(Path("analysis/results").glob("*")):
    print(f"  {f.name}")
