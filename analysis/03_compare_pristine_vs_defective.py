"""Compare pristine vs defective MOF-303 RDFs from MD at 150K.

Creates side-by-side and overlay plots to highlight defect signatures.
"""

import numpy as np
import matplotlib.pyplot as plt

# Load pre-computed RDFs
r_p, g_p = np.loadtxt("analysis/rdf_150K_pristine.csv", unpack=True)
r_d, g_d = np.loadtxt("analysis/rdf_150K_defective.csv", unpack=True)

# ===== 1. Overlay plot =====
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(r_p, g_p, "b-", linewidth=1.5, label="Pristine MOF-303")
ax.plot(r_d, g_d, "r-", linewidth=1.5, label="Defective MOF-303 (Br)")
ax.set_xlabel("r (Å)", fontsize=12)
ax.set_ylabel("g(r)", fontsize=12)
ax.set_title("RDF Comparison — Pristine vs Defective MOF-303 at 150K", fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, 10)
plt.tight_layout()
plt.savefig("analysis/rdf_comparison_overlay.png", dpi=150)
plt.close()

# ===== 2. Difference plot =====
# Interpolate to common grid if needed
r_common = np.linspace(0, 10, 500)
g_p_interp = np.interp(r_common, r_p, g_p)
g_d_interp = np.interp(r_common, r_d, g_d)
diff = g_d_interp - g_p_interp

fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})

axes[0].plot(r_common, g_p_interp, "b-", linewidth=1.5, label="Pristine")
axes[0].plot(r_common, g_d_interp, "r-", linewidth=1.5, label="Defective")
axes[0].set_ylabel("g(r)", fontsize=12)
axes[0].set_title("RDF — Pristine vs Defective MOF-303 at 150K", fontsize=13)
axes[0].legend(fontsize=11)
axes[0].set_xlim(0, 10)

axes[1].fill_between(r_common, diff, 0, where=diff > 0, alpha=0.3, color="red", label="Defective > Pristine")
axes[1].fill_between(r_common, diff, 0, where=diff < 0, alpha=0.3, color="blue", label="Pristine > Defective")
axes[1].plot(r_common, diff, "k-", linewidth=1)
axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.5)
axes[1].set_xlabel("r (Å)", fontsize=12)
axes[1].set_ylabel("Δg(r)", fontsize=12)
axes[1].set_title("Difference: g_defective(r) - g_pristine(r)", fontsize=11)
axes[1].legend(fontsize=10)
axes[1].set_xlim(0, 10)

plt.tight_layout()
plt.savefig("analysis/rdf_comparison_difference.png", dpi=150)
plt.close()

# ===== 3. Partial RDF comparison (Al-O) =====
try:
    r_p_alo, g_p_alo = np.loadtxt("analysis/rdf_150K_pristine_Al_O.csv", unpack=True)
    r_d_alo, g_d_alo = np.loadtxt("analysis/rdf_150K_defective_Al_O.csv", unpack=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r_p_alo, g_p_alo, "b-", linewidth=1.5, label="Pristine Al-O")
    ax.plot(r_d_alo, g_d_alo, "r-", linewidth=1.5, label="Defective Al-O")
    ax.set_xlabel("r (Å)", fontsize=12)
    ax.set_ylabel("g(r)", fontsize=12)
    ax.set_title("Al-O Partial RDF — Effect of Br Defects on Al Coordination", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 6)
    plt.tight_layout()
    plt.savefig("analysis/rdf_comparison_Al_O.png", dpi=150)
    plt.close()
except Exception as e:
    print(f"Skipping Al-O comparison: {e}")

print("Plots saved:")
print("  analysis/rdf_comparison_overlay.png")
print("  analysis/rdf_comparison_difference.png")
print("  analysis/rdf_comparison_Al_O.png")
