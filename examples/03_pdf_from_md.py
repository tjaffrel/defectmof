"""Example: Compute PDF from MD trajectory.

Computes g(r) using ASE's built-in RDF (quick check).
For experimental comparison, use compute_pdf() with diffpy engine.
"""
from defectmof import compute_rdf

r, g_r = compute_rdf(
    snapshots="md_150K_pristine.traj",
    rmax=10.0,
    nbins=200,
)

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(r, g_r)
plt.xlabel("r (A)")
plt.ylabel("g(r)")
plt.title("RDF from MD at 150K — Pristine MOF-303")
plt.savefig("rdf_150K_pristine.png")
print("Saved rdf_150K_pristine.png")
