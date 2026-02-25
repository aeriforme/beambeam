#!/usr/bin/env python3
"""
Written by Anna Kinderman 02/23/26

Convert GuineaPig++ pairs0.dat (ASCII) to openPMD (HDF5) particle file.

Assumed file format (first 7 columns):
  E_GeV  beta_x  beta_y  beta_z  x_nm  y_nm  z_nm  [ignored extra columns...]

Conventions:
- sign(E_GeV) encodes species:
    E>0 -> electron, E<0 -> positron
- beta components are v/c (direction + speed)

Momentum reconstruction:
--momentum total   : assumes |E| is total energy (GeV); p = beta * sqrt(E^2 - m^2)

Output openPMD:
- positions in meters
- momenta in SI (kg*m/s) with correct unitSI
- weight = 1.0 for each macro-particle (override with --weight-scale if desired)
"""

import argparse
import math
import numpy as np

try:
    import openpmd_api as io

#In case there are installation or dependency errors
except ImportError as e:
    raise SystemExit(
        "Missing dependency openpmd_api. Install with:\n"
        "  conda install -c conda-forge openpmd-api\n"
        "or:\n"
        "  pip install openPMD-api\n"
    ) from e


# Physical constants
C = 299_792_458.0  # m/s
ELECTRON_MASS_GEV = 0.00051099895000  # GeV (mc^2)
GEV_C_TO_SI = 5.344286e-19  # (GeV/c) -> (kg*m/s)

#Let's step through the GP file
def parse_pairs_ascii(path: str):
    """
    Returns:
      electrons: dict of arrays {E, bx, by, bz, x_nm, y_nm, z_nm}
      positrons: same
    """
    # Load only first 7 columns; ignore trailing numbers
    data = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            try:
                vals = [float(parts[i]) for i in range(7)]
            except ValueError:
                continue
            data.append(vals)

    if not data:
        raise ValueError(f"No valid particle rows found in {path}")

    arr = np.asarray(data, dtype=np.float64)
    E = arr[:, 0]
    bx, by, bz = arr[:, 1], arr[:, 2], arr[:, 3]
    x_nm, y_nm, z_nm = arr[:, 4], arr[:, 5], arr[:, 6]

    #Distinguish particle species based on sign
    ele_mask = E > 0
    pos_mask = E < 0
    
    #Conversion to particle species, of which we only need electrons and positrons
    electrons = dict(E=np.abs(E[ele_mask]), bx=bx[ele_mask], by=by[ele_mask], bz=bz[ele_mask],
                     x_nm=x_nm[ele_mask], y_nm=y_nm[ele_mask], z_nm=z_nm[ele_mask])

    positrons = dict(E=np.abs(E[pos_mask]), bx=bx[pos_mask], by=by[pos_mask], bz=bz[pos_mask],
                     x_nm=x_nm[pos_mask], y_nm=y_nm[pos_mask], z_nm=z_nm[pos_mask])

    return electrons, positrons


def momentum_gev_c(particles: dict):
    """
    Compute p components in GeV/c.
    """
    E = particles["E"]
    bx, by, bz = particles["bx"], particles["by"], particles["bz"]
    
    #magnitude of beta
    beta_mag = np.sqrt(bx*bx + by*by + bz*bz)
    beta_mag = np.where(beta_mag > 0, beta_mag, 1.0)  # avoid divide-by-zero
    nx, ny, nz = bx/beta_mag, by/beta_mag, bz/beta_mag

    #E is total energy in GeV (includes rest mass), p = sqrt(E^2 - m^2)
    p_mag = np.sqrt(np.maximum(E*E - ELECTRON_MASS_GEV**2, 0.0))
    px, py, pz = nx*p_mag, ny*p_mag, nz*p_mag
    return px, py, pz


def write_openpmd(out_path: str, electrons: dict, positrons: dict, weight_scale: float,
                  iteration: int = 0):
    series = io.Series(out_path, io.Access.create)
    series.set_software("gp_pairs_to_openpmd.py", "1.0")
    # openPMD standard metadata is handled by openPMD-api; schema conventions are defined by the openPMD standard. :contentReference[oaicite:2]{index=2}

    it = series.iterations[iteration]
    it.time = 0.0
    it.dt = 1.0

    def add_species(species_name: str, parts: dict):
        n = len(parts["E"])
        if n == 0:
            return

        species = it.particles[species_name]
        species.set_attribute("unitDimension", np.array([0., 0., 0., 0., 0., 0., 0.], dtype=np.float64))

        # positions: nm -> m since openPMD is all in SI
        x = parts["x_nm"] * 1e-9
        y = parts["y_nm"] * 1e-9
        z = parts["z_nm"] * 1e-9

        # momenta in GeV/c -> SI kg*m/s
        px_g, py_g, pz_g = momentum_gev_c(parts)
        px = px_g * GEV_C_TO_SI
        py = py_g * GEV_C_TO_SI
        pz = pz_g * GEV_C_TO_SI

        w = np.full(n, weight_scale, dtype=np.float64)

        # Create resizable datasets
        # openPMD-api provides standard ways to write particle records. :contentReference[oaicite:3]{index=3}
        for comp, arr_m in [("x", x), ("y", y), ("z", z)]:
            rec = species["position"][comp]
            rec.reset_dataset(io.Dataset(arr_m.dtype, extent=[n]))
            rec.store_chunk(arr_m, offset=[0], extent=[n])
            rec.unit_SI = 1.0  # meters

        for comp, arr_p in [("x", px), ("y", py), ("z", pz)]:
            rec = species["momentum"][comp]
            rec.reset_dataset(io.Dataset(arr_p.dtype, extent=[n]))
            rec.store_chunk(arr_p, offset=[0], extent=[n])
            rec.unit_SI = 1.0  # already SI

        recw = species["weighting"][io.Record_Component.SCALAR]
        recw.reset_dataset(io.Dataset(w.dtype, extent=[n]))
        recw.store_chunk(w, offset=[0], extent=[n])
        recw.unit_SI = 1.0

    add_species("electrons", electrons)
    add_species("positrons", positrons)

    it.close()
    series.flush()
    series.close()


def main():
    ap = argparse.ArgumentParser(description="Convert GuineaPig pairs0.dat to openPMD-HDF5 particle file.")
    ap.add_argument("pairs_dat", help="Input pairs0.dat (ASCII)")
    ap.add_argument("-o", "--out", default="gp_pairs.openpmd.h5", help="Output openPMD file (HDF5)")
    ap.add_argument("--weight-scale", type=float, default=1.0,
                    help="Write particle weight = this value for all particles (default 1.0)")
    ap.add_argument("--iteration", type=int, default=0, help="Iteration index to write (default 0)")
    args = ap.parse_args()

    electrons, positrons = parse_pairs_ascii(args.pairs_dat)
    write_openpmd(args.out, electrons, positron,
                 weight_scale=args.weight_scale,
                 iteration=args.iteration)

    ne = len(electrons["E"])
    np_ = len(positrons["E"])
    print(f"Wrote {args.out}")
    print(f"  electrons : {ne}")
    print(f"  positrons : {np_}")
    print(f"  weight    : {args.weight_scale}")


if __name__ == "__main__":
    main()
