# Szigorúbb galaktikus vágás (|b|>30°) saját maszkkal

# make_latcut_mask.py
import sys, numpy as np
from astropy.io import fits
import ducc0.healpix as hp0

if len(sys.argv) != 4:
    print("Használat: python make_latcut_mask.py <input_mask.fits> <Bdeg> <output_mask.fits>")
    sys.exit(1)

inp, bdeg_str, outp = sys.argv[1], sys.argv[2], sys.argv[3]
bdeg = float(bdeg_str)

# Beolvasás
m = fits.getdata(inp)
hdr = fits.getheader(inp)

# Maszk binarizálása + lapítás
m = np.asarray(m, dtype=float).ravel()
nside = int(hdr.get("NSIDE", 2048))
npix = 12 * nside * nside
if m.size != npix:
    raise ValueError(f"Input maszk hossza ({m.size}) nem egyezik a NSIDE szerinti npix-szel ({npix}).")

m_bin = (m > 0.5).astype(float)

# Galaktikus szélesség (b) előállítása a HEALPix pixelekhez
hp = hp0.Healpix_Base(nside, "RING")
ang = hp.pix2ang(np.arange(npix, dtype=np.int64))  # (npix, 2) tömb
theta = ang[:, 0]                                  # kolatitud (0..pi)
# b = 90° - theta°
b_deg = 90.0 - np.degrees(theta)

# |b| >= Bdeg vágás
latcut = (np.abs(b_deg) >= bdeg).astype(float)

# Metszet az eredeti maszkkal
m_out = (m_bin * latcut).astype(np.float32)

# HEADER tisztítás / beállítás (ORDERING legyen RING)
hdr = hdr.copy()
hdr["ORDERING"] = "RING"
hdr["NSIDE"] = nside
hdr["PIXTYPE"] = "HEALPIX"

# Mentés
hdu = fits.PrimaryHDU(m_out, header=hdr)
hdu.writeto(outp, overwrite=True)

# Gyors visszajelzés
masked_frac = float((m_out == 0).mean())
print(f"Saved strict mask: {outp}  |  masked fraction = {masked_frac:.3f}  (|b| >= {bdeg:.1f}°)")
