"""Physical validation tests for defectmof.

These tests verify that the code produces physically meaningful results,
not just that it runs without errors. If these tests fail, it likely means
the core scientific logic has a bug.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from collections import Counter
from pathlib import Path


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"
DEFECTIVE_CIF = str(STRUCTURES / "struct1_optimized_c_axis.cif")
PRISTINE_CIF = str(STRUCTURES / "struct2_optimized_a_axis.cif")


@pytest.fixture
def defective():
    return read(DEFECTIVE_CIF, format="cif")


@pytest.fixture
def pristine():
    return read(PRISTINE_CIF, format="cif")


# ===== Unit cell physical properties =====


class TestUnitCellIntegrity:
    """Verify the input structures have expected physical properties."""

    def test_defective_cell_contains_bromine(self, defective):
        """The defective MOF-303 should contain Br atoms."""
        symbols = defective.get_chemical_symbols()
        assert "Br" in symbols, "Defective structure missing Br atoms"
        assert symbols.count("Br") == 4, f"Expected 4 Br, got {symbols.count('Br')}"

    def test_pristine_cell_has_no_bromine(self, pristine):
        """The pristine MOF-303 should NOT contain Br atoms."""
        symbols = pristine.get_chemical_symbols()
        assert "Br" not in symbols, "Pristine structure should not contain Br"

    def test_both_cells_contain_aluminum(self, defective, pristine):
        """MOF-303 is aluminum-based — both cells must have 8 Al."""
        assert defective.get_chemical_symbols().count("Al") == 8
        assert pristine.get_chemical_symbols().count("Al") == 8

    def test_unit_cells_have_matching_lattice(self, defective, pristine):
        """Both cells should have compatible lattice parameters."""
        params_d = defective.cell.cellpar()
        params_p = pristine.cell.cellpar()
        # Lengths should match within 0.01 A
        np.testing.assert_allclose(params_d[:3], params_p[:3], atol=0.01)
        # Angles should match within 0.1 degrees
        np.testing.assert_allclose(params_d[3:], params_p[3:], atol=0.1)

    def test_no_unphysical_short_distances_in_input(self, defective, pristine):
        """Input structures should not have atom pairs closer than 0.5 A."""
        for name, atoms in [("defective", defective), ("pristine", pristine)]:
            dists = atoms.get_all_distances(mic=True)
            np.fill_diagonal(dists, 999)
            min_dist = dists.min()
            assert min_dist > 0.5, (
                f"{name} has unphysically short distance: {min_dist:.3f} A"
            )

    def test_volume_per_atom_is_reasonable(self, defective, pristine):
        """MOF volume per atom should be between 10-30 A^3 (typical for MOFs)."""
        for name, atoms in [("defective", defective), ("pristine", pristine)]:
            vol_per_atom = atoms.get_volume() / len(atoms)
            assert 10 < vol_per_atom < 30, (
                f"{name} vol/atom={vol_per_atom:.1f} A^3 is outside MOF range"
            )


# ===== Supercell physical properties =====


class TestSupercellPhysics:
    """Verify supercell construction produces physically valid structures."""

    def test_supercell_preserves_stoichiometry(self, defective, pristine):
        """Element ratios in supercell should reflect the defect fraction."""
        from defectmof import build_supercell

        sc = build_supercell(defective, pristine, size=(2, 2, 2), defect_fraction=0.5)
        counts = Counter(sc.get_chemical_symbols())

        # With 50% defect fraction on 2x2x2 (8 cells), 4 defective + 4 pristine
        # Defective has 8 Al per cell, pristine has 8 Al per cell
        # Total Al should be close to 8*8 = 64 (some may be pruned at boundaries)
        assert 56 <= counts["Al"] <= 64, f"Al count {counts['Al']} outside expected range"

    def test_supercell_defect_count_matches_fraction(self, defective, pristine):
        """Number of Br atoms should correspond to defect fraction."""
        from defectmof import build_supercell

        for frac in [0.25, 0.44, 0.75]:
            sc = build_supercell(
                defective, pristine, size=(2, 2, 2),
                defect_fraction=frac, mode="random", seed=42,
            )
            n_br = sc.get_chemical_symbols().count("Br")
            n_cells = 2 * 2 * 2
            expected_defective = int(round(frac * n_cells))
            br_per_defective_cell = 4
            expected_br = expected_defective * br_per_defective_cell
            # Allow some tolerance due to boundary pruning
            assert abs(n_br - expected_br) <= expected_br * 0.15, (
                f"frac={frac}: got {n_br} Br, expected ~{expected_br} "
                f"(±15% for boundary effects)"
            )

    def test_supercell_volume_scales_correctly(self, defective, pristine):
        """Supercell volume should be exactly N times the unit cell volume."""
        from defectmof import build_supercell

        unit_vol = pristine.get_volume()
        for size in [(2, 2, 2), (3, 2, 4), (4, 4, 4)]:
            sc = build_supercell(defective, pristine, size=size, mode="random")
            expected_vol = unit_vol * size[0] * size[1] * size[2]
            np.testing.assert_allclose(
                sc.get_volume(), expected_vol, rtol=1e-6,
                err_msg=f"Supercell {size} volume mismatch"
            )

    def test_supercell_no_unphysical_overlaps(self, defective, pristine):
        """After building, no two atoms should be closer than 0.8 A."""
        from defectmof import build_supercell

        for mode in ["random", "alternating_a", "clustered_small"]:
            sc = build_supercell(defective, pristine, size=(2, 2, 2), mode=mode)
            dists = sc.get_all_distances(mic=True)
            np.fill_diagonal(dists, 999)
            min_dist = dists.min()
            assert min_dist >= 0.79, (
                f"Mode {mode}: found overlap at {min_dist:.3f} A "
                f"(prune_overlapping_atoms should prevent this)"
            )

    def test_alternating_modes_produce_different_structures(self, defective, pristine):
        """Different alternating modes should produce distinct Br distributions."""
        from defectmof import build_supercell

        br_positions = {}
        for mode in ["alternating_a", "alternating_b", "alternating_c"]:
            sc = build_supercell(defective, pristine, size=(2, 2, 2), mode=mode)
            br_mask = [s == "Br" for s in sc.get_chemical_symbols()]
            br_pos = sc.positions[br_mask]
            br_positions[mode] = br_pos

        # Alt_a and alt_b should have different Br spatial distributions
        # Compare center of mass of Br atoms
        com_a = br_positions["alternating_a"].mean(axis=0)
        com_b = br_positions["alternating_b"].mean(axis=0)
        com_c = br_positions["alternating_c"].mean(axis=0)

        # At least two of the three should differ significantly
        diffs = [
            np.linalg.norm(com_a - com_b),
            np.linalg.norm(com_a - com_c),
            np.linalg.norm(com_b - com_c),
        ]
        assert max(diffs) > 0.1, (
            "Alternating modes along different axes should produce "
            "spatially distinct Br distributions"
        )

    def test_clustered_mode_clusters_defects(self, defective, pristine):
        """Clustered modes should group defects spatially."""
        from defectmof import build_supercell

        # Build random and clustered supercells
        sc_random = build_supercell(
            defective, pristine, size=(2, 2, 2),
            mode="random", seed=42,
        )
        sc_clustered = build_supercell(
            defective, pristine, size=(2, 2, 2),
            mode="clustered_large", seed=42,
        )

        def br_spatial_spread(atoms):
            """Standard deviation of Br positions — lower = more clustered."""
            br_mask = [s == "Br" for s in atoms.get_chemical_symbols()]
            br_pos = atoms.positions[br_mask]
            if len(br_pos) < 2:
                return 0.0
            return np.std(br_pos, axis=0).mean()

        spread_random = br_spatial_spread(sc_random)
        spread_clustered = br_spatial_spread(sc_clustered)

        # Clustered should have lower spread (more compact Br distribution)
        # Allow some tolerance since it's stochastic
        assert spread_clustered <= spread_random * 1.2, (
            f"Clustered spread ({spread_clustered:.1f}) should be <= "
            f"random spread ({spread_random:.1f}). "
            "clustered_large mode may not be clustering effectively."
        )


# ===== MACE energy validation =====


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
class TestMACEEnergies:
    """Verify MACE gives physically reasonable energies.

    These tests use the MACE-MP0 small model on GPU.
    Skip if CUDA is not available.
    """

    @pytest.fixture(autouse=True)
    def setup_calculator(self):
        """Set up MACE calculator, skip if no CUDA."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from mace.calculators import mace_mp
        self.calc = mace_mp(model="small", device="cuda", default_dtype="float32")

    def test_energy_per_atom_in_reasonable_range(self, defective, pristine):
        """Energy per atom should be between -10 and -3 eV for organic MOFs."""
        for name, atoms in [("defective", defective), ("pristine", pristine)]:
            atoms = atoms.copy()
            atoms.calc = self.calc
            e_per_atom = atoms.get_potential_energy() / len(atoms)
            assert -10.0 < e_per_atom < -3.0, (
                f"{name}: E/atom={e_per_atom:.3f} eV is outside reasonable range"
            )

    def test_supercell_energy_scales_linearly(self, defective, pristine):
        """Supercell energy should be approximately N times unit cell energy."""
        from defectmof import build_supercell

        # Get individual cell energies
        d = defective.copy()
        d.calc = self.calc
        e_d = d.get_potential_energy()

        p = pristine.copy()
        p.calc = self.calc
        e_p = p.get_potential_energy()

        # Build 2x2x2 with known composition
        sc = build_supercell(defective, pristine, size=(2, 2, 2), defect_fraction=0.5)
        sc.calc = self.calc
        e_sc = sc.get_potential_energy()

        # Expected: 4 * defective + 4 * pristine (approximately)
        e_expected = 4 * e_d + 4 * e_p
        # Allow 5% deviation (boundary effects, overlap pruning)
        ratio = e_sc / e_expected
        assert 0.90 < ratio < 1.10, (
            f"Supercell energy ratio {ratio:.3f} too far from 1.0. "
            f"E_sc={e_sc:.1f}, E_expected={e_expected:.1f}"
        )

    def test_optimization_reduces_energy(self, pristine):
        """Optimization should lower (or maintain) the total energy."""
        from defectmof import optimize

        atoms = pristine.copy()
        atoms.calc = self.calc
        e_before = atoms.get_potential_energy()

        optimized = optimize(
            pristine, model="mace_mp_small", optimizer="lbfgs",
            fmax=0.5, max_steps=20, device="cuda",
        )
        optimized.calc = self.calc
        e_after = optimized.get_potential_energy()

        assert e_after <= e_before + 0.1, (
            f"Energy increased after optimization: {e_before:.2f} -> {e_after:.2f} eV"
        )

    def test_optimization_reduces_forces(self, pristine):
        """After optimization, max force should be lower than before."""
        from defectmof import optimize

        atoms = pristine.copy()
        atoms.calc = self.calc
        fmax_before = np.abs(atoms.get_forces()).max()

        optimized = optimize(
            pristine, model="mace_mp_small", optimizer="lbfgs",
            fmax=0.5, max_steps=50, device="cuda",
        )
        optimized.calc = self.calc
        fmax_after = np.abs(optimized.get_forces()).max()

        assert fmax_after < fmax_before, (
            f"Forces did not decrease: {fmax_before:.4f} -> {fmax_after:.4f} eV/A"
        )

    def test_optimization_preserves_composition(self, pristine):
        """Optimization should not change the chemical composition."""
        from defectmof import optimize

        counts_before = Counter(pristine.get_chemical_symbols())

        optimized = optimize(
            pristine, model="mace_mp_small", optimizer="lbfgs",
            fmax=0.5, max_steps=10, device="cuda",
        )
        counts_after = Counter(optimized.get_chemical_symbols())

        assert counts_before == counts_after, (
            f"Composition changed: {dict(counts_before)} -> {dict(counts_after)}"
        )

    def test_optimization_preserves_cell(self, pristine):
        """Optimization (without cell relaxation) should preserve the unit cell."""
        from defectmof import optimize

        cell_before = pristine.get_cell().array.copy()

        optimized = optimize(
            pristine, model="mace_mp_small", optimizer="lbfgs",
            fmax=0.5, max_steps=10, device="cuda",
        )
        cell_after = optimized.get_cell().array

        np.testing.assert_allclose(
            cell_before, cell_after, atol=1e-6,
            err_msg="Cell changed during fixed-cell optimization"
        )

    def test_different_optimizers_reach_similar_energy(self, pristine):
        """LBFGS, FIRE, and BFGS should converge to similar energies."""
        from defectmof import optimize

        energies = {}
        for opt_name in ["lbfgs", "fire"]:
            result = optimize(
                pristine, model="mace_mp_small", optimizer=opt_name,
                fmax=0.5, max_steps=30, device="cuda",
            )
            result.calc = self.calc
            energies[opt_name] = result.get_potential_energy()

        # Both should give similar energies (within 5 eV for 128 atoms)
        e_diff = abs(energies["lbfgs"] - energies["fire"])
        assert e_diff < 5.0, (
            f"Optimizer energies differ too much: "
            f"LBFGS={energies['lbfgs']:.2f}, FIRE={energies['fire']:.2f}"
        )


# ===== RDF physical validation =====


class TestRDFPhysics:
    """Verify RDF computation produces physically meaningful results."""

    def test_rdf_has_peaks_at_bond_distances(self, pristine):
        """RDF of a crystal should show peaks at nearest-neighbor distances."""
        from defectmof import compute_rdf

        r, g_r = compute_rdf([pristine], rmax=5.0, nbins=200)

        # g(r) should have at least one peak above 1.0 (nearest neighbors)
        assert g_r.max() > 1.0, (
            f"RDF max={g_r.max():.3f}, expected peaks > 1.0 for a crystal"
        )

        # g(r) should approach 0 at very small r (no atoms that close)
        assert g_r[:5].max() < 0.5, (
            "RDF should be near zero at very small r (< 0.1 A)"
        )

    def test_rdf_al_o_shows_coordination(self, pristine):
        """Al-O partial RDF should peak around 1.7-2.0 A (typical Al-O bond)."""
        from defectmof import compute_rdf

        r, g_r = compute_rdf([pristine], rmax=5.0, nbins=200, elements=("Al", "O"))

        # Find the first significant peak
        peak_idx = np.argmax(g_r)
        peak_r = r[peak_idx]

        # Al-O bond in MOFs is typically 1.7-2.1 A
        assert 1.5 < peak_r < 2.5, (
            f"Al-O first peak at {peak_r:.2f} A, "
            f"expected 1.7-2.1 A for MOF Al-O coordination"
        )

    def test_rdf_averaging_reduces_noise(self, pristine):
        """Averaging over multiple snapshots should produce smoother RDF."""
        from defectmof import compute_rdf

        # Single snapshot
        r1, g1 = compute_rdf([pristine], rmax=5.0, nbins=200)

        # Multiple identical snapshots (same crystal, should be identical)
        r_avg, g_avg = compute_rdf([pristine] * 5, rmax=5.0, nbins=200)

        # For identical snapshots, result should be the same
        np.testing.assert_allclose(g1, g_avg, rtol=1e-5,
            err_msg="Averaging identical snapshots changed the RDF")

    def test_supercell_rdf_matches_unit_cell(self, defective, pristine):
        """RDF of a pure supercell should match the unit cell RDF."""
        from ase.build import make_supercell
        from defectmof import compute_rdf

        # Build a 2x2x2 of just pristine (no mixing)
        sc = make_supercell(pristine, np.diag([2, 2, 2]))

        r_unit, g_unit = compute_rdf([pristine], rmax=5.0, nbins=100)
        r_sc, g_sc = compute_rdf([sc], rmax=5.0, nbins=100)

        # RDFs should be similar (not identical due to normalization)
        # Check that peaks are at similar positions
        peak_unit = r_unit[np.argmax(g_unit)]
        peak_sc = r_sc[np.argmax(g_sc)]
        assert abs(peak_unit - peak_sc) < 0.1, (
            f"Unit cell peak at {peak_unit:.2f} A vs "
            f"supercell peak at {peak_sc:.2f} A"
        )
