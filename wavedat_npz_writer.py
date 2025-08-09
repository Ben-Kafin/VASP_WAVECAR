#!/usr/bin/env python3
"""
wavedat_npz_writer.py

Parses WAVECAR + EIGENVAL, caches to .npz, and optionally
saves k-point–weighted averaged coefficients and energies.
If only one k-point is present, averaging is skipped.
"""

import os
import numpy as np
from pymatgen.io.vasp.outputs import Wavecar


class WavecarEigenvalParser:
    """
    Parses WAVECAR + EIGENVAL, extracts:
      • coeffs:   complex ndarray (nspins, nk, nb, nG)
      • kpts:     float ndarray (nk, 3)
      • weights:  float ndarray (nk,)
      • energies: float ndarray (nspins, nk, nb)

    Can save a binary cache (.npz) of these arrays, and optionally
    include k-weighted averages over k-points for both coeffs
    and energies. Averaging is skipped if only one k-point.
    """

    def __init__(self, directory: str, tol: float = 1e-6):
        self.directory   = os.path.abspath(directory)
        self.wavecar_pth = os.path.join(self.directory, "WAVECAR")
        self.eigenval_pth= os.path.join(self.directory, "EIGENVAL")
        self.cache_pth   = os.path.join(self.directory, "wavedat_cache.npz")
        self.tol         = tol

        # to be filled by parse() or loaded from cache
        self.coeffs   = None    # (nspins, nk, nb, nG)
        self.kpts     = None    # (nk, 3)
        self.weights  = None    # (nk,)
        self.energies = None    # (nspins, nk, nb)

    def parse(self):
        """
        If cache exists, load it. Otherwise:
          1) read WAVECAR → self.coeffs
          2) parse EIGENVAL → self.kpts, self.weights, self.energies
        """
        if os.path.exists(self.cache_pth):
            data = np.load(self.cache_pth, allow_pickle=True)
            self.coeffs   = data["coeffs"]
            self.kpts     = data["kpts"]
            self.weights  = data["weights"]
            self.energies = data["energies"]
            return

        # 1) Read WAVECAR coefficients
        if not os.path.isfile(self.wavecar_pth):
            raise FileNotFoundError(f"WAVECAR not found at {self.wavecar_pth}")
        wave = Wavecar(self.wavecar_pth)
        self.coeffs = np.array(wave.coeffs)

        # 2) Parse k-points, weights, and band energies from EIGENVAL
        self.kpts, self.weights, self.energies = self._parse_eigenval()

    def summary(self):
        """Print parsed‐data dimensions and weights."""
        nspins, nk, nb, nG = self.coeffs.shape
        print("\nParsed Files Summary")
        print("--------------------")
        print(f" Directory      : {self.directory}")
        print(f" Spin channels  : {nspins}")
        print(f" k-points       : {nk}")
        print(f" Bands          : {nb}")
        print(f" G-vectors      : {nG}")
        print(f" k-point weights: {self.weights.tolist()}")
        print(f" Energies shape : {self.energies.shape}\n")

    def average_over_kpoints(self):
        """
        Collapse the k-dimension of coeffs:
          avg_coeffs[spin, band, g] =
            ∑ₖ wₖ·coeffs[spin,k,band,g] / ∑ₖ wₖ
        Returns ndarray (nspins, nb, nG).
        """
        w     = np.array(self.weights, float)
        wsum  = w.sum()
        if wsum == 0:
            raise ValueError("Sum of k-point weights is zero.")
        w    /= wsum
        coeffs = np.array(self.coeffs)
        return np.tensordot(coeffs, w, axes=([1], [0]))

    def average_energies(self):
        """
        Collapse the k-dimension of raw energies:
          avg_e[spin, band] =
            ∑ₖ wₖ·energies[spin,k,band] / ∑ₖ wₖ
        Returns ndarray (nspins, nb).
        """
        w     = np.array(self.weights, float)
        wsum  = w.sum()
        if wsum == 0:
            raise ValueError("Sum of k-point weights is zero.")
        w    /= wsum
        e     = np.array(self.energies)
        return np.tensordot(e, w, axes=([1], [0]))

    def save_cache(self, weighted_average: bool = False):
        """
        Save raw arrays and optionally averaged coeffs+energies to .npz.
        If weighted_average=True but only one k-point, averaging is skipped.
        """
        arrays = {
            "coeffs":   self.coeffs,
            "kpts":     np.array(self.kpts),
            "weights":  np.array(self.weights),
            "energies": self.energies,
        }

        nk = len(self.weights)
        if weighted_average and nk > 1:
            arrays["avg_coeffs"]   = self.average_over_kpoints()
            arrays["avg_energies"] = self.average_energies()
        elif weighted_average:
            print("Only one k-point; skipping averaged coeffs & energies.")

        np.savez(self.cache_pth, **arrays)
        keys = ", ".join(arrays.keys())
        print(f"Binary cache saved to {self.cache_pth} (keys: {keys})")

    def run(self, weighted_average: bool = False):
        """
        parse → summary → save_cache(weighted_average)
        Returns: (coeffs, kpts, weights, energies)
        """
        self.parse()
        self.summary()
        self.save_cache(weighted_average)
        return self.coeffs, self.kpts, self.weights, self.energies

    def _parse_eigenval(self):
        """
        Parse VASP EIGENVAL to extract k-point coords, weights, and band energies.
        Returns:
          kpts     : list of length nk, each a (3,) tuple
          weights  : list of length nk
          energies : ndarray (nspins, nk, nb)
        Assumes:
          - 6 header lines at the top.
          - For each k-point block:
              • one or more blank lines
              • one line:    kx ky kz weight    (all in sci notation)
              • nb lines:    band# E_up  E_dn  p1  p2
        """
        if not os.path.isfile(self.eigenval_pth):
            raise FileNotFoundError(f"EIGENVAL not found at {self.eigenval_pth}")

        with open(self.eigenval_pth) as f:
            lines = f.readlines()

        # Fetch shapes from WAVECAR coeffs
        nspins, nk, nb, _ = self.coeffs.shape

        kpts    = []
        weights = []
        energies= np.zeros((nspins, nk, nb), dtype=float)

        idx = 6  # skip the 6‐line header
        for ik in range(nk):
            # skip blank lines
            while idx < len(lines) and not lines[idx].strip():
                idx += 1

            # kx, ky, kz, weight line
            tok = lines[idx].split()
            kx, ky, kz = map(float, tok[:3])
            w          = float(tok[3])
            kpts.append((kx, ky, kz))
            weights.append(w)
            idx += 1

            # next nb lines: band idx, spin-up E, spin-down E, p1, p2
            for ib in range(nb):
                tok = lines[idx].split()
                energies[0, ik, ib] = float(tok[1])
                if nspins > 1:
                    energies[1, ik, ib] = float(tok[2])
                idx += 1

        return np.array(kpts), np.array(weights), energies


if __name__ == "__main__":
    # Example one-click run (Spyder-friendly)
    directory = r'C:/Users/Benjamin Kafin/Documents/VASP/fcc/NHC/ncl'
    parser    = WavecarEigenvalParser(directory)
    parser.run(weighted_average=True)