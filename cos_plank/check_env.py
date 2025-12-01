# check_env.py
import sys
ok = True
try:
    import numpy as np; print("numpy:", np.__version__)
    import astropy, astropy.io.fits as fits; print("astropy:", astropy.__version__)
    import ducc0, ducc0.healpix as hp; print("ducc0:", ducc0.__version__)
    print("ducc0.healpix.map2alm available:", hasattr(hp, "map2alm"))
except Exception as e:
    ok = False; print("Import error:", e)

try:
    import healpy as hp; print("healpy: OK")
except Exception:
    print("healpy: NOT available (this is fine if ducc0.healpix.map2alm is True)")

if not ok:
    sys.exit(1)
