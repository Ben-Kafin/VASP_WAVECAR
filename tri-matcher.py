# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

from wavedat_npz_writer import WavecarEigenvalParser


class WaveMatcher:
    """
    Match bands between two component systems (simple, metal) and a full system,
    using k-resolved wavefunctions and energies from .npz caches.

    For each full band:
      - find best simple match and best metal match by |<component|full>|^2 at a chosen k-point
      - optionally pick a single "primary" match (larger overlap) to color the full-state line
      - always show both matches in the interactive hover; bold the chosen one if enabled

    Plot layout:
      - Metal (top), Simple (middle), Full (bottom)
      - Independent color "centers" for metal and simple panels

    Outputs:
      - band_matches_combined.txt (one line per full band with both simple and metal matches)
      - ortho_band_matches.txt (full bands where both overlaps <= ortho_tol)

    Energy alignment:
      - Full: shifted by its Fermi level
      - Metal: shifted by its Fermi level (no bottom alignment)
      - Simple: bottom-aligned to Full (per spin)
    """

    def __init__(self, simple_dir, metal_dir, full_dir,
                 ortho_tol=1e-8, tol=1e-6):
        self.simple_dir = simple_dir
        self.metal_dir  = metal_dir
        self.full_dir   = full_dir
        self.ortho_tol  = ortho_tol
        self.tol        = tol

        # k-resolved arrays
        self.coeffs_s = None   # (nspins, nk, nb, nG)
        self.ener_s   = None   # (nspins, nk, nb)
        self.w_s      = None   # (nk,)

        self.coeffs_m = None
        self.ener_m   = None
        self.w_m      = None

        self.coeffs_f = None
        self.ener_f   = None
        self.w_f      = None

        # k-averaged energies (for plotting)
        self.e_s = None       # (nspins, nb)
        self.e_m = None
        self.e_f = None

        # Records
        self.matches = []         # non-orthogonal records per full band
        self.ortho_matches = []   # both overlaps <= ortho_tol

    @staticmethod
    def _wavefunction_norm(vec):
        """
        Normalize a wavefunction with respect to the Hermitian inner product.
        Uses sqrt(<ψ|ψ>) so ||ψ||=1 (keeps complex phase intact).
        """
        flat = np.asarray(vec).ravel()
        nrm2 = float(np.vdot(flat, flat).real)
        if nrm2 <= 0:
            return flat
        return flat / np.sqrt(nrm2)

    def get_fermi_level_from_doscar(self, directory):
        """Reads DOSCAR and returns the Fermi energy (4th token, line 6)."""
        path = os.path.join(directory, "DOSCAR")
        with open(path, "r") as f:
            lines = f.readlines()
        if len(lines) < 6:
            raise ValueError(f"DOSCAR must have at least 6 lines: {path}")
        return float(lines[5].split()[3])

    def load_from_cache(self, directory):
        """
        Ensure wavedat_cache.npz exists. If missing, generate it with
        k-weight averaging OFF. Returns raw arrays:
          coeffs   : (nspins, nk, nb, nG)
          energies : (nspins, nk, nb)
          weights  : (nk,)
        """
        cache_path = os.path.join(directory, "wavedat_cache.npz")
        if not os.path.exists(cache_path):
            print(f"Cache not found at {cache_path}; generating now...")
            parser = WavecarEigenvalParser(directory, tol=self.tol)
            parser.run(weighted_average=False)

        data     = np.load(cache_path, allow_pickle=True)
        coeffs   = data["coeffs"]        # (nspins, nk, nb, nG), complex
        energies = data["energies"]      # (nspins, nk, nb), float
        weights  = data["weights"].astype(float)  # (nk,), float
        return coeffs, energies, weights

    def _kavg_energies(self, E, w):
        """Return k-weighted average energies: (nspins, nb)."""
        w = np.array(w, float)
        w = w / w.sum()
        return np.tensordot(E, w, axes=([1], [0]))

    def match_bands(self, use_k_index=0):
        """
        Vectorized pipeline:
          1) Load simple/metal/full coeffs & energies (+ raw weights)
          2) Use Γ-point energies for plotting/shift (as in your current code)
          3) Shift full by EF(full), metal by EF(metal); bottom-align simple to full
          4) Vectorized overlaps at k-point index = use_k_index:
               O_s = S_norm @ F_norm.conj().T   -> shape (nb_s, nb_f)
               O_m = M_norm @ F_norm.conj().T   -> shape (nb_m, nb_f)
             Then argmax along rows to find best match per full band.
        """
        # Load caches (unchanged)
        if self.coeffs_s is None:
            self.coeffs_s, self.ener_s, w_s = self.load_from_cache(self.simple_dir)
            self.w_s = w_s / w_s.sum()
        if self.coeffs_m is None:
            self.coeffs_m, self.ener_m, w_m = self.load_from_cache(self.metal_dir)
            self.w_m = w_m / w_m.sum()
        if self.coeffs_f is None:
            self.coeffs_f, self.ener_f, w_f = self.load_from_cache(self.full_dir)
            self.w_f = w_f / w_f.sum()
    
        # Shapes & checks
        nspins, nk_s, nb_s, nG_s = self.coeffs_s.shape
        nspins_m, nk_m, nb_m, nG_m = self.coeffs_m.shape
        nspins_f, nk_f, nb_f, nG_f = self.coeffs_f.shape
        if not (nspins == nspins_m == nspins_f):
            raise ValueError("Spin channel count mismatch among systems.")
        if not (nk_s == nk_m == nk_f):
            raise ValueError("Number of k-points mismatch among systems.")
        if use_k_index < 0 or use_k_index >= nk_f:
            raise ValueError(f"use_k_index out of range: 0..{nk_f-1}")
        if not (nG_s == nG_m == nG_f):
            raise ValueError("Plane-wave dimension (nG) mismatch among systems.")
    
        # Γ-point energies (no k-averaging), as you currently do
        self.e_s = self.ener_s[:, 0, :]
        self.e_m = self.ener_m[:, 0, :]
        self.e_f = self.ener_f[:, 0, :]
    
        # Shift by EF: full & metal
        ef_full  = self.get_fermi_level_from_doscar(self.full_dir)
        ef_metal = self.get_fermi_level_from_doscar(self.metal_dir)
        self.e_f -= ef_full
        self.e_m -= ef_metal
    
        # Bottom-align simple to full (per spin); do NOT bottom-align metal
        for s in range(nspins):
            min_f = self.e_f[s].min()
            self.e_s[s] -= (self.e_s[s].min() - min_f)
    
        # Clear
        self.matches.clear()
        self.ortho_matches.clear()
    
        # Core: vectorized overlaps per spin at selected k-index
        ik = use_k_index
    
        # Small helper to normalize rows: (nb, nG) -> (nb, nG)
        def normalize_rows(mat):
            # mat: (nb, nG) complex
            # norm = sqrt(real(vdot(row,row))) per row
            # Avoid division by zero
            norms = np.sqrt(np.maximum(np.einsum('ij,ij->i', mat.conj(), mat).real, 1e-300))
            return (mat.T / norms).T
    
        for spin in range(nspins):
            # Extract and reshape to (nb, nG), ensure contiguous complex128
            F = np.ascontiguousarray(self.coeffs_f[spin, ik, :, :].reshape(nb_f, nG_f))
            S = np.ascontiguousarray(self.coeffs_s[spin, ik, :, :].reshape(nb_s, nG_s))
            M = np.ascontiguousarray(self.coeffs_m[spin, ik, :, :].reshape(nb_m, nG_m))
    
            # Normalize once per band
            Fn = normalize_rows(F)
            Sn = normalize_rows(S)
            Mn = normalize_rows(M)
    
            # All-to-all overlaps via BLAS (complex GEMM)
            # shapes: (nb_s, nG) @ (nG, nb_f) -> (nb_s, nb_f)
            O_s = Sn @ Fn.conj().T
            O_m = Mn @ Fn.conj().T
    
            Ov_s = np.abs(O_s) ** 2  # (nb_s, nb_f)
            Ov_m = np.abs(O_m) ** 2  # (nb_m, nb_f)
    
            # Best matches per full band (argmax over component bands)
            best_s_idx = Ov_s.argmax(axis=0)           # (nb_f,)
            best_s_ov  = Ov_s[best_s_idx, np.arange(nb_f)]
            best_m_idx = Ov_m.argmax(axis=0)           # (nb_f,)
            best_m_ov  = Ov_m[best_m_idx, np.arange(nb_f)]
    
            # Build records
            for ib_f in range(nb_f):
                si = int(best_s_idx[ib_f])
                mi = int(best_m_idx[ib_f])
    
                rec = {
                    "spin": spin,
                    "full_idx": ib_f + 1,
                    "E_full":   float(self.e_f[spin, ib_f]),
                    "simple": {
                        "idx":     si + 1,
                        "E":       float(self.e_s[spin, si]),
                        "overlap": float(best_s_ov[ib_f]),
                        "dE":      float(self.e_f[spin, ib_f] - self.e_s[spin, si]),
                    },
                    "metal": {
                        "idx":     mi + 1,
                        "E":       float(self.e_m[spin, mi]),
                        "overlap": float(best_m_ov[ib_f]),
                        "dE":      float(self.e_f[spin, ib_f] - self.e_m[spin, mi]),
                    },
                }
    
                if (rec["simple"]["overlap"] <= self.ortho_tol) and (rec["metal"]["overlap"] <= self.ortho_tol):
                    self.ortho_matches.append(rec)
                else:
                    self.matches.append(rec)

    def write_results(self):
        """
        Write:
          - band_matches_combined.txt (sorted by full_idx, then spin)
            Each line includes both simple and metal matches for a full state.
          - ortho_band_matches.txt (cases where both overlaps <= ortho_tol)
        """
        out_c = os.path.join(self.full_dir, "band_matches_combined.txt")
        out_o = os.path.join(self.full_dir, "ortho_band_matches.txt")

        combined = sorted(self.matches, key=lambda r: (r["full_idx"], r["spin"]))
        orthos   = sorted(self.ortho_matches, key=lambda r: (r["full_idx"], r["spin"]))

        header_c = (
            "# band_matches_combined.txt\n"
            "# One line per full band (non-orthogonal), sorted by full_idx.\n"
            "# spin full_idx  E_full(eV)  |  Simple: idx  E(eV)  |ov|^2  dE(eV)  |  Metal: idx  E(eV)  |ov|^2  dE(eV)\n"
        )
        with open(out_c, "w") as f:
            f.write(header_c)
            for r in combined:
                s = r["simple"]
                m = r["metal"]
                f.write(
                    f"{r['spin']:2d}  {r['full_idx']:6d}  {r['E_full']:10.3f}  |  "
                    f"{s['idx']:6d}  {s['E']:9.3f}  {s['overlap']:8.6f}  {s['dE']:8.3f}  |  "
                    f"{m['idx']:6d}  {m['E']:9.3f}  {m['overlap']:8.6f}  {m['dE']:8.3f}\n"
                )

        header_o = "# ortho_band_matches.txt\n# spin full_idx  E_full(eV)  (both overlaps <= ortho_tol)\n"
        with open(out_o, "w") as f:
            f.write(header_o)
            for r in orthos:
                f.write(f"{r['spin']:2d}  {r['full_idx']:6d}  {r['E_full']:10.3f}\n")

        print(f"Wrote {len(combined)} combined matches to {out_c}")
        print(f"Wrote {len(orthos)} orthogonal cases to {out_o}")

    def final_plot(self,
                   energy_range=None,
                   center_simple=None,
                   center_metal=None,
                   cmap_name_simple="coolwarm",
                   cmap_name_metal="berlin",
                   power_simple=1.0,
                   power_metal=0.175,
                   pick_primary=True,
                   interactive=False):
        if not self.matches and not self.ortho_matches:
            raise RuntimeError("No matches computed. Call match_bands() first.")
    
        def build_colors(values, matched_indices, center_idx, cmap, power):
            arr = np.array(values)
            order = sorted(enumerate(arr, start=1), key=lambda x: x[1])
            N = len(order)
    
            if center_idx is not None and N > 1:
                pivot_pos = next((i for i, (idx, _) in enumerate(order) if idx == center_idx), N // 2)
            else:
                pivot_pos = N // 2
    
            max_left = pivot_pos
            max_right = N - pivot_pos - 1
            denom = max(max_left, max_right)
    
            colors = {}
            for i, (idx, _) in enumerate(order):
                signed = (i - pivot_pos) / denom if denom > 0 else 0.0
                val = 0.5 + 0.5 * np.sign(signed) * (abs(signed) ** power)
                colors[idx] = cmap(np.clip(val, 0, 1)) if idx in matched_indices else "black"
            return colors
    
        def build_colors_energy_scaled(values, matched_indices, center_idx, cmap, power):
            arr = np.array(values)
            order = sorted(enumerate(arr, start=1), key=lambda x: x[1])
            energies = np.array([v for _, v in order])
    
            center_energy = next((v for idx, v in order if idx == center_idx), np.median(energies))
            max_dist = max(abs(energies[0] - center_energy), abs(energies[-1] - center_energy))
    
            colors = {}
            for i, (idx, energy) in enumerate(order):
                signed = (energy - center_energy) / max_dist if max_dist > 0 else 0.0
                val = 0.5 + 0.5 * np.sign(signed) * (abs(signed) ** power)
                colors[idx] = cmap(np.clip(val, 0, 1)) if idx in matched_indices else "black"
            return colors
    
        s = 0
        cmap_simple = plt.get_cmap(cmap_name_simple)
        cmap_metal  = plt.get_cmap(cmap_name_metal)
    
        recs = [r for r in self.matches if r["spin"] == s]
        recs += [r for r in self.ortho_matches if r["spin"] == s]
        recs = sorted(recs, key=lambda r: r["full_idx"])
    
        primary_simple = set()
        secondary_simple = set()
        primary_metal = set()
        secondary_metal = set()
    
        for r in recs:
            s_ov, m_ov = r["simple"]["overlap"], r["metal"]["overlap"]
            s_idx, m_idx = r["simple"]["idx"], r["metal"]["idx"]
            s_matched = s_ov > self.ortho_tol
            m_matched = m_ov > self.ortho_tol
    
            if pick_primary and (s_matched or m_matched):
                if s_ov >= m_ov:
                    primary_simple.add(s_idx)
                    if m_matched:
                        secondary_metal.add(m_idx)
                else:
                    primary_metal.add(m_idx)
                    if s_matched:
                        secondary_simple.add(s_idx)
            else:
                if s_matched:
                    primary_simple.add(s_idx)
                if m_matched:
                    primary_metal.add(m_idx)
    
        if not pick_primary:
            secondary_simple.clear()
            secondary_metal.clear()
    
        all_simple = set(range(1, len(self.e_s[s]) + 1))
        all_metal  = set(range(1, len(self.e_m[s]) + 1))
    
        colors_simple = build_colors(self.e_s[s], all_simple, center_simple, cmap_simple, power_simple)
        colors_metal  = build_colors_energy_scaled(self.e_m[s], all_metal, center_metal, cmap_metal, power_metal)
    
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        ax_m, ax_s, ax_f = axes
    
        for ax in axes:
            ax.axvline(0, color="k", ls=":")
    
        for idx in all_metal:
            E = self.e_m[s][idx - 1]
            if idx in primary_metal:
                ls, c = "-", colors_metal[idx]
            elif idx in secondary_metal:
                ls, c = "--", colors_metal[idx]
            else:
                ls, c = "--", "black"
            ax_m.vlines(E, 0, 1, color=c, linestyle=ls, lw=2)
        ax_m.set_title("Metal")
        ax_m.set_ylabel("Normalized")
    
        for idx in all_simple:
            E = self.e_s[s][idx - 1]
            if idx in primary_simple:
                ls, c = "-", colors_simple[idx]
            elif idx in secondary_simple:
                ls, c = "--", colors_simple[idx]
            else:
                ls, c = "--", "black"
            ax_s.vlines(E, 0, 1, color=c, linestyle=ls, lw=2)
        ax_s.set_title("Simple")
        ax_s.set_ylabel("Normalized")
    
        artists = []
        for r in recs:
            s_ov = r["simple"]["overlap"]
            m_ov = r["metal"]["overlap"]
            s_idx = r["simple"]["idx"]
            m_idx = r["metal"]["idx"]
            E = r["E_full"]
    
            s_matched = s_ov > self.ortho_tol
            m_matched = m_ov > self.ortho_tol
    
            if not s_matched and not m_matched:
                line = ax_f.vlines(E, 0, 1, color="black", linestyle="--", lw=2)
            elif pick_primary:
                c = colors_simple.get(s_idx) if s_ov >= m_ov else colors_metal.get(m_idx)
                line = ax_f.vlines(E, 0, 1, color=c, linestyle="-", lw=2)
            else:
                c = colors_simple.get(s_idx) if s_matched else colors_metal.get(m_idx, "black")
                line = ax_f.vlines(E, 0, 1, color=c, linestyle="-", lw=2)
            artists.append(line)
    
        ax_f.set_title("Full")
        ax_f.set_ylabel("Normalized")
        ax_f.set_xlabel("Energy (eV)")
    
        if energy_range:
            for ax in axes:
                ax.set_xlim(energy_range)
    
        plt.tight_layout()
    
        if interactive and artists:
            cursor = mplcursors.cursor(artists, hover=True)
    
            def _math_line(label, idx, E, ov, dE, bold=False):
                txt = (
                    rf"\mathrm{{{label}}}\ \#{{{idx}}},\ "
                    rf"E={E:.3f}\ \mathrm{{eV}},\ "
                    rf"\left|ov\right|^2={ov:.6f},\ "
                    rf"\Delta E={dE:.3f}\ \mathrm{{eV}}"
                )
                return rf"${{\mathbf{{{txt}}}}}$" if bold else rf"${{{txt}}}$"
    
            @cursor.connect("add")
            def on_add(sel):
                i = artists.index(sel.artist)
                r = recs[i]
                primary_simple = r["simple"]["overlap"] >= r["metal"]["overlap"]
                s_bold = bool(pick_primary and primary_simple)
                m_bold = bool(pick_primary and not primary_simple)
    
                header = (
                    rf"$\mathrm{{Full}}\ \#{{{r['full_idx']}}},\ "
                    rf"E={r['E_full']:.3f}\ \mathrm{{eV}}$"
                )
                s_line = _math_line("Simple", r["simple"]["idx"],
                                    r["simple"]["E"], r["simple"]["overlap"], r["simple"]["dE"],
                                    bold=s_bold)
                m_line = _math_line("Metal", r["metal"]["idx"],
                                    r["metal"]["E"], r["metal"]["overlap"], r["metal"]["dE"],
                                    bold=m_bold)
                sel.annotation.set_text(f"{header}\n{s_line}\n{m_line}")
    
        plt.show()

    def run(self,
            energy_range=None,
            center_simple=None,
            center_metal=None,
            cmap_name_simple="coolwarm",
            cmap_name_metal="berlin",
            power_simple=1.0,
            power_metal=0.175,
            pick_primary=True,
            interactive=True,
            use_k_index=0):
        """
        Full pipeline: match → write → plot.
        Now supports separate colormap/power settings per system.
        """
        self.match_bands(use_k_index=use_k_index)
        self.write_results()
        self.final_plot(
            energy_range=energy_range,
            center_simple=center_simple,
            center_metal=center_metal,
            cmap_name_simple=cmap_name_simple,
            cmap_name_metal=cmap_name_metal,
            power_simple=power_simple,
            power_metal=power_metal,
            pick_primary=pick_primary,
            interactive=interactive
        )
        
if __name__ == "__main__":
    # --- EDIT THESE THREE DIRECTORIES ---
    # Use raw strings (r"...") for Windows paths.
    simple_dir = r"dir_component1"
    metal_dir  = r"dir_component2"  # <-- set your metal component path
    full_dir   = r"dir_fullsystem"
    # ------------------------------------



    matcher = WaveMatcher(simple_dir, metal_dir, full_dir)

    matcher.run(
        energy_range=(-25, 10),
        center_simple=40,
        center_metal=602,
        cmap_name_simple="coolwarm",
        cmap_name_metal="berlin",
        power_simple=0.36,
        power_metal=0.075,
        pick_primary=True,
        interactive=True,
        use_k_index=0
    )