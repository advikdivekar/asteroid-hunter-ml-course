# Milestone 1: Inject one fake asteroid into one ZTF frame and verify it looks real.

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import galsim

# Config
FITS_FILE = '/Users/advikdivekar/Desktop/asteroid/ztf_test/sci/2018/0325/263518/ztf_20180325263518_000618_zg_c03_o_q4_sciimg.fits'
TARGET_MAG  = 19.0     # fake asteroid brightness (try 18, 19, 20, 21)
INJECT_X    = 1500.7   # sub-pixel x position (pixels, 0-indexed)
INJECT_Y    = 1200.3   # sub-pixel y position
STAMP_SIZE  = 21       # pixels; must be odd
# 

def load_image(fits_path):
    """Load ZTF FITS and pull the values we need from the header."""
    hdul = fits.open(fits_path)
    data   = hdul[0].data.astype(np.float64)   # must be float for math
    header = hdul[0].header
    hdul.close()

    magzp    = header['MAGZP']      # zeropoint: converts flux → magnitude
    seeing   = header['SEEING']     # FWHM of atmospheric blur, arcseconds
    pixscale = header['PIXSCALE']   # arcseconds per pixel (ZTF ≈ 1.012)
    gain     = header['GAIN']       # electrons per ADU (ZTF ≈ 6.2)

    print(f"MAGZP    = {magzp:.3f}   (zeropoint)")
    print(f"SEEING   = {seeing:.3f} arcsec FWHM")
    print(f"PIXSCALE = {pixscale:.4f} arcsec/pixel")
    print(f"GAIN     = {gain:.2f} e-/ADU")
    return data, magzp, seeing, pixscale, gain


def inject_asteroid(data, magzp, seeing, pixscale, gain,
                    mag, x_pix, y_pix, stamp_size=21):
    """
    Inject a realistic point source at (x_pix, y_pix) with magnitude mag.

    Steps:
      1. Convert magnitude → flux in ADU using MAGZP
      2. Build PSF (Moffat profile — more accurate than Gaussian for ZTF)
      3. Convolve a delta function (point source) with the PSF
      4. Draw a small stamp image with sub-pixel offset
      5. Add Poisson shot noise to the stamp (makes it statistically real)
      6. Paste the stamp into the full image

    Returns modified image (copy) and the stamp array.
    """
    # 1. Flux 
    # flux = 10^(-(mag - magzp) / 2.5)
    # This is the AB magnitude system: smaller mag = more flux
    flux = 10 ** (-(mag - magzp) / 2.5)
    print(f"\nInjecting mag={mag:.1f}  →  flux={flux:.1f} ADU")

    #2. PSF 
    # Moffat profile: β=4.5 is standard for ground-based seeing.
    # fwhm must be in PIXELS (seeing is in arcsec, divide by pixscale)
    fwhm_pixels = seeing / pixscale
    print(f"PSF FWHM = {fwhm_pixels:.3f} pixels")
    psf = galsim.Moffat(beta=4.5, fwhm=fwhm_pixels)

    # 3. Convolve delta function with PSF  
    # A real star = delta function (infinitely sharp) blurred by the PSF.
    # galsim.DeltaFunction(flux) is that idealized point source.
    point  = galsim.DeltaFunction(flux=flux)
    source = galsim.Convolve([point, psf])

    # 4. Draw stamp with sub-pixel offset
    # Integer center of stamp in the full image
    x_int = int(round(x_pix))
    y_int = int(round(y_pix))
    # Fractional offset from integer position (this is the key sub-pixel step)
    dx = x_pix - x_int
    dy = y_pix - y_int

    # drawImage returns a galsim.Image; .array gives us a numpy array
    galsim_stamp = source.drawImage(
        nx=stamp_size,
        ny=stamp_size,
        scale=1.0,                         # 1 GalSim unit = 1 pixel
        offset=galsim.PositionD(dx, dy)    # sub-pixel shift
    )
    stamp = galsim_stamp.array.copy()

    # 5. Poisson noise
    # Real sources have shot noise: variance = signal (in electrons).
    # Convert stamp to electrons, sample Poisson, convert back to ADU.
    stamp_electrons = stamp * gain
    # np.random.poisson needs non-negative input; clip any tiny negatives
    stamp_electrons = np.maximum(stamp_electrons, 0)
    noisy_electrons = np.random.poisson(stamp_electrons).astype(np.float64)
    noisy_stamp     = noisy_electrons / gain
    # The noise we ADD is (noisy - noiseless); keeps total flux correct
    noise_only = noisy_stamp - stamp
    stamp_final = stamp + noise_only

    # 6. Paste into full image
    half = stamp_size // 2
    y0 = y_int - half
    x0 = x_int - half

    # Boundary check — make sure stamp doesn't go off the edge
    assert y0 >= 0 and x0 >= 0, "Injection position too close to image edge!"
    assert y0 + stamp_size <= data.shape[0], "Stamp extends past bottom edge!"
    assert x0 + stamp_size <= data.shape[1], "Stamp extends past right edge!"

    modified = data.copy()   # NEVER modify the original image
    modified[y0:y0+stamp_size, x0:x0+stamp_size] += stamp_final

    return modified, stamp_final


def verify_injection(original, modified, x_int, y_int, mag, window=60):
    """
    Four-panel plot:
      Panel 1: Original image around injection site
      Panel 2: Modified image (with fake asteroid)
      Panel 3: Difference (what we added)
      Panel 4: Real stars nearby for PSF comparison
    """
    half = window // 2
    y0, y1 = y_int - half, y_int + half
    x0, x1 = x_int - half, x_int + half

    orig_crop  = original [y0:y1, x0:x1]
    mod_crop   = modified [y0:y1, x0:x1]
    diff_crop  = mod_crop - orig_crop

    # Use same stretch for before/after so you can compare brightness
    vmin = np.nanpercentile(orig_crop, 5)
    vmax = np.nanpercentile(orig_crop, 99)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Injection Verification  (mag={mag:.1f}, '
                 f'x={x_int}, y={y_int})', fontsize=14)

    ax = axes[0]
    ax.imshow(orig_crop, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title('Before — original image')
    ax.set_xlabel('pixels')

    ax = axes[1]
    ax.imshow(mod_crop, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    # Red cross at injection centre
    cx, cy = half, half
    ax.axhline(cy, color='red', linewidth=0.7, alpha=0.6)
    ax.axvline(cx, color='red', linewidth=0.7, alpha=0.6)
    ax.set_title(f'After — fake asteroid injected')
    ax.set_xlabel('pixels')

    ax = axes[2]
    im = ax.imshow(diff_crop, cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax, label='ADU added')
    ax.set_title('Difference (what we added)\nShould look like a star PSF')
    ax.set_xlabel('pixels')

    plt.tight_layout()
    plt.savefig('milestone1_injection.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: milestone1_injection.png")
    print("\nQUICK CHECK:")
    print(f"  Peak flux added : {diff_crop.max():.1f} ADU")
    print(f"  Image noise RMS : {np.std(orig_crop):.1f} ADU")
    print(f"  SNR estimate    : {diff_crop.max() / np.std(orig_crop):.1f}")
    print("\nIf SNR > 5 → easily detectable. If 3-5 → faint but real. If < 3 → too faint.")
    print("If the difference panel looks like a star (round blob) → injection is correct!")


# MAIN 
if __name__ == '__main__':
    np.random.seed(42)   # reproducible noise

    print("=" * 60)
    print("MILESTONE 1: Synthetic Injection")
    print("=" * 60)

    # Load
    original, magzp, seeing, pixscale, gain = load_image(FITS_FILE)
    print(f"\nImage shape: {original.shape}  (rows × cols)")

    # Inject
    modified, stamp = inject_asteroid(
        original, magzp, seeing, pixscale, gain,
        mag=TARGET_MAG,
        x_pix=INJECT_X,
        y_pix=INJECT_Y,
        stamp_size=STAMP_SIZE
    )

    # Verify
    x_int = int(round(INJECT_X))
    y_int = int(round(INJECT_Y))
    verify_injection(original, modified, x_int, y_int, TARGET_MAG)

    print("\nMilestone 1 complete.")
    print("Try changing TARGET_MAG to 20, 21 and re-run — watch the SNR drop.")