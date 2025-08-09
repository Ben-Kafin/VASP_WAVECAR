# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 18:52:41 2025

@author: Benjamin Kafin
"""

#!/usr/bin/env python3
"""
wave_matcher.py

Band-matching using k-point–resolved wavefunctions.  Each overlap
is computed on unit-normalized complex coefficient vectors *per k-point*,
then collapsed via a weighted average of the overlap magnitudes.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

from wavedat_npz_writer import WavecarEigenvalParser


class WaveMatcher:
    """
    Match bands between a simple and a full system using
    k-resolved wavefunctions and energies loaded directly
    from a .npz cache (auto-generated if missing, with
    k-weight averaging turned OFF). Overlaps are computed
    per k-point then collapsed by weighting each complex
    overlap by its magnitude.
    """

    def __init__(self, simple_dir, full_dir, ortho_tol=1e-8, tol=1e-6):
        self.simple_dir = simple_dir
        self.full_dir   = full_dir
        self.ortho_tol  = ortho_tol
        self.tol        = tol

        # k-resolved arrays
        self.coeffs_s = None   # (nspins, nk, nb, nG)
        self.ener_s   = None   # (nspins, nk, nb)
        self.w_s      = None   # (nk,)

        self.coeffs_f = None
        self.ener_f   = None
        self.w_f      = None

        # k-averaged energies
        self.e_s = None       # (nspins, nb)
        self.e_f = None

        self.matches       = []
        self.ortho_matches = []

    @staticmethod
    def _wavefunction_norm(vec):
        """
        Normalize a vector by the absolute value squared of a dot product with itself
        """
        flat = vec.ravel()
        norm = np.vdot(flat,flat)
        nvec = flat / np.sqrt(norm)
        return nvec


    def get_fermi_level_from_doscar(self, directory):
        """
        Reads DOSCAR and returns the Fermi energy (4th token, line 6).
        """
        path = os.path.join(directory, "DOSCAR")
        with open(path, "r") as f:
            lines = f.readlines()
        if len(lines) < 6:
            raise ValueError("DOSCAR must have at least 6 lines.")
        return float(lines[5].split()[3])

    def load_from_cache(self, directory):
        """
        Ensure wavedat_cache.npz exists. If missing, generate it
        with k-weight averaging OFF. Returns raw arrays:
          coeffs   : (nspins, nk, nb, nG)
          energies : (nspins, nk, nb)
          weights  : (nk,)
        """
        cache_path = os.path.join(directory, "wavedat_cache.npz")

        if not os.path.exists(cache_path):
            print(f"Cache not found at {cache_path}; generating now...")
            parser = WavecarEigenvalParser(directory, tol=self.tol)
            # disable k-weight averaging in the writer
            parser.run(weighted_average=False)

        data     = np.load(cache_path, allow_pickle=True)
        coeffs   = data["coeffs"]    # (nspins, nk, nb, nG)
        energies = data["energies"]  # (nspins, nk, nb)
        weights  = data["weights"].astype(float)  # (nk,)

        return coeffs, energies, weights

    def match_bands(self):
        """
        1) Load simple/full k-point-resolved coeffs & energies
        2) Compute k-averaged energies for shifting & bottom-align
        3) For each (spin, full-band) find the best simple band by:
             a) computing complex overlap at each k-point
             b) picking the simple band with largest magnitude overlap 
                (abs(complex overlap)^2)
        """
        # load or retrieve full k-resolved data + raw weights
        if self.coeffs_s is None:
            self.coeffs_s, self.ener_s, w_s = self.load_from_cache(self.simple_dir)
            self.w_s = w_s / w_s.sum()
        if self.coeffs_f is None:
            self.coeffs_f, self.ener_f, w_f = self.load_from_cache(self.full_dir)
            self.w_f = w_f / w_f.sum()

        # build k-averaged band energies (for shifting & bottom-align)
        self.e_s = np.tensordot(self.ener_s, self.w_s, axes=([1], [0]))
        self.e_f = np.tensordot(self.ener_f, self.w_f, axes=([1], [0]))

        # shift full energies by Fermi level
        ef_full = self.get_fermi_level_from_doscar(self.full_dir)
        self.e_f -= ef_full

        ef_simple = self.get_fermi_level_from_doscar(self.simple_dir)
        self.e_s -= ef_simple
        """
        # bottom-align simple → full
        delta = self.e_s.min() - self.e_f.min()
        self.e_s -= delta
        """

        # unpack shapes
        nspins, nk_s, nb_s, nG = self.coeffs_s.shape
        _,      nk_f, nb_f, _  = self.coeffs_f.shape
        if nk_s != nk_f:
            raise ValueError("Number of k-points in simple vs full mismatch.")

        # clear any previous runs
        self.matches.clear()
        self.ortho_matches.clear()
        
        for spin in range(nspins):
            for ib_f in range(nb_f):
                best_ib, best_ov = None, 0.0
        
                # loop over simple bands, but only use the first k-point (ik=0)
                for ib_s in range(nb_s):
                    v_f = self.coeffs_f[spin, 0, ib_f]
                    v_s = self.coeffs_s[spin, 0, ib_s]
                    
                    ov_local = np.vdot(
                        self._wavefunction_norm(v_s),
                        self._wavefunction_norm(v_f)
                    )
                    
                    #ov_local = np.vdot(v_s, v_f)
                    # magnitude squared is now a scalar float
                    ov_glob = (np.abs(ov_local))**2
        
                    if best_ib is None or ov_glob > best_ov:
                        best_ib, best_ov = ib_s, ov_glob
        
                # record result
                dE = self.e_f[spin, ib_f] - self.e_s[spin, best_ib]
                rec = {
                    "spin":       spin,
                    "simple_idx": best_ib + 1,
                    "full_idx":   ib_f   + 1,
                    "E_simple":   self.e_s[spin, best_ib],
                    "E_full":     self.e_f[spin, ib_f],
                    "overlap":    best_ov,
                    "dE":         dE
                }
        
                # categorize into orthogonal vs matched
                if best_ov <= self.ortho_tol:
                    self.ortho_matches.append(rec)
                else:
                    self.matches.append(rec)

    def write_results(self):
        """
        Write band_matches.txt and ortho_band_matches.txt in full_dir,
        sorting by simple band index and printing only the overlap magnitude.
        """
        out1 = os.path.join(self.simple_dir, "band_matches_norm.txt")
        out2 = os.path.join(self.simple_dir, "ortho_band_matches.txt")
    
        # sort matches by simple band index
        sorted_matches = sorted(self.matches, key=lambda r: (r["simple_idx"], -r["overlap"]))
        sorted_ortho  = sorted(self.ortho_matches, key=lambda r: r["simple_idx"])
    
        header1 = "# spin simple_idx full_idx E_simple(eV) E_full(eV) ov_mag dE\n"
        with open(out1, "w") as f:
            f.write(header1)
            for r in sorted_matches:
                f.write(
                    f"{r['spin']:2d}   "
                    f"{r['simple_idx']:3d}   "
                    f"{r['full_idx']:3d}   "
                    f"{r['E_simple']:8.3f}   "
                    f"{r['E_full']:8.3f}   "
                    f"{r['overlap']:7.6f}   "
                    f"{r['dE']:7.3f}\n"
                )
    
        header2 = "# spin simple_idx full_idx E_simple(eV) E_full(eV) overlap=0\n"
        with open(out2, "w") as f:
            f.write(header2)
            for r in sorted_ortho:
                # overlap always zero for ortho matches
                f.write(
                    f"{r['spin']:2d}   "
                    f"{r['simple_idx']:3d}   "
                    f"{r['full_idx']:3d}   "
                    f"{r['E_simple']:8.3f}   "
                    f"{r['E_full']:8.3f}   "
                    "0.000\n"
                )
    
        print(f"Wrote {len(sorted_matches)} matches to {out1}")
        print(f"Wrote {len(sorted_ortho)} orthogonal matches to {out2}")

    def final_plot(self,
                   energy_range=None,
                   center_seq=None,
                   cmap_name="coolwarm",
                   power=1.0,
                   interactive=False):
        """
        Plot simple/full energies as vertical lines, colored by match.
        """
        E_s = self.e_s[0]

        sorted_simple = sorted(enumerate(E_s, start=1), key=lambda x: x[1])
        N             = len(sorted_simple)
        cmap          = plt.get_cmap(cmap_name)

        pivot = ((center_seq-1)/(N-1)) if center_seq else 0.5
        half  = max(pivot, 1-pivot)
        colors = {
            idx: cmap(np.clip(
                    0.5 + 0.5*np.sign(((pos-1)/(N-1)-pivot)/half)
                         * abs(((pos-1)/(N-1)-pivot)/half)**power,
                    0,1))
            for pos,(idx,_) in enumerate(sorted_simple, start=1)
        }

        matched = {r["simple_idx"] for r in self.matches}

        fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(8,6))
        ax1.axvline(0, color="k", ls=":")
        ax2.axvline(0, color="k", ls=":")

        for idx, e in sorted_simple:
            c  = colors[idx] if idx in matched else "black"
            ls = "-" if idx in matched else "--"
            ax1.vlines(e, 0, 1, color=c, linestyle=ls, lw=2)
        ax1.set_title("Simple")
        ax1.set_ylabel("Normalized")

        artists = []
        for r in sorted(self.matches, key=lambda x: x["full_idx"]):
            line = ax2.vlines(r["E_full"], 0, 1,
                              color=colors[r["simple_idx"]], lw=2)
            artists.append(line)
        ax2.set_title("Full")
        ax2.set_ylabel("Normalized")
        ax2.set_xlabel("Energy (eV)")

        if energy_range:
            ax1.set_xlim(energy_range)
            ax2.set_xlim(energy_range)
        plt.tight_layout()

        if interactive and artists:
            cursor = mplcursors.cursor(artists, hover=True)
            @cursor.connect("add")
            def on_add(sel):
                i = artists.index(sel.artist)
                r = sorted(self.matches, key=lambda x: x["full_idx"])[i]
                sel.annotation.set_text(
                    f"Full #{r['full_idx']} @ {r['E_full']:.3f} eV\n"
                    f"Simple #{r['simple_idx']} @ {r['E_simple']:.3f} eV\n"
                    f"Overlap {r['overlap']:.6f}\n"
                    f"dE {r['dE']:.3f} eV"
                )

        plt.show()

    def run(self,
            energy_range=None,
            center_seq=None,
            cmap_name="coolwarm",
            power=1.0,
            interactive=False):
        """
        Full pipeline: match → write → plot.
        """
        self.match_bands()
        self.write_results()
        self.final_plot(
            energy_range=energy_range,
            center_seq=center_seq,
            cmap_name=cmap_name,
            power=power,
            interactive=interactive
        )


if __name__ == "__main__":
    simple = r"dir1"
    full   = r"dir2"

    matcher = WaveMatcher(simple, full)
    matcher.run(
        energy_range=(-25, 10),
        center_seq=602,
        cmap_name="coolwarm",
        power=0.36,
        interactive=True

    )
