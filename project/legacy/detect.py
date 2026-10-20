# Milestone 2: Detect all sources in an injected image and confirm. SEP recovers the fake asteroid.

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sep

# Re-use the injection function from inject.py
import sys
sys.path.insert(0, '/Users/advikdivekar/Desktop/asteroid')
from inject import load_image, inject_asteroid

# CONFIG 
FITS_FILE   = '/Users/advikdivekar/Desktop/asteroid/ztf_test/sci/2018/0325/263518/ztf_20180325263518_000618_zg_c03_o_q4_sciimg.fits'
TARGET_MAG  = 19.0
INJECT_X    = 1500.7
INJECT_Y    = 1200.3
DETECT_THRESH = 6.0    # sigma threshold — 3σ catches faint objects
MATCH_RADIUS  = 6.0    # pixels — how close counts as a "match"

def detect_sources(image_data, thresh=3.0):
    """
    Detect all sources in an image using SEP.

    SEP works in 3 steps:
      1. Estimate the sky background (slowly varying large-scale structure)
      2. Subtract it so we're left with sources on a flat zero baseline
      3. Find connected blobs of pixels above thresh × background RMS

    Returns a numpy structured array with fields:
      x, y       — centroid position (pixels, 0-indexed)
      flux       — total flux in ADU (approximate)
      peak       — peak pixel value
      a, b       — semi-axes of ellipse fit
      theta      — orientation angle
    """
    # SEP requires C-contiguous float32 array
    data = image_data.astype(np.float32)
    data = np.ascontiguousarray(data)

    # Background estimation
    bkg = sep.Background(data)
    data_sub = data - bkg.back()   # sky-subtracted image

    # Source extraction
    sources = sep.extract(data_sub, thresh=thresh, err=bkg.rms())

    # FWHM from the ellipse axes (SEP reports a, b in pixels)
    # FWHM ≈ 2 * sqrt(2*ln2) * sigma ≈ 2.355 * (a+b)/2
    sources_fwhm = 2.355 * (sources['a'] + sources['b']) / 2.0

    return sources, sources_fwhm, bkg


def find_match(sources, x_target, y_target, radius=3.0):
    """
    Search the source catalog for a detection within `radius` pixels
    of (x_target, y_target). Returns the matching row or None.
    """
    dist = np.sqrt((sources['x'] - x_target)**2 +
                   (sources['y'] - y_target)**2)
    idx = np.argmin(dist)
    if dist[idx] < radius:
        return idx, dist[idx]
    return None, None


def plot_detection(image_sub, sources, x_inj, y_inj, match_idx, window=80):
    """
    Plot the sky-subtracted image around the injection site,
    with all detected sources shown as blue ellipses and
    the recovered asteroid shown in red.
    """
    x_int = int(round(x_inj))
    y_int = int(round(y_inj))
    half  = window // 2
    y0, y1 = y_int - half, y_int + half
    x0, x1 = x_int - half, x_int + half

    crop = image_sub[y0:y1, x0:x1]
    vmin = np.nanpercentile(crop, 5)
    vmax = np.nanpercentile(crop, 99)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(crop, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(f'SEP detections around injection site\n'
                 f'Blue = all sources  |  Red = recovered fake asteroid')

    # Draw all sources in the window as blue ellipses
    for i, s in enumerate(sources):
        sx, sy = s['x'] - x0, s['y'] - y0
        if 0 < sx < window and 0 < sy < window:
            from matplotlib.patches import Ellipse
            e = Ellipse(
                xy=(sx, sy),
                width  = 6 * s['a'],
                height = 6 * s['b'],
                angle  = np.degrees(s['theta']),
                edgecolor='cyan', facecolor='none', linewidth=1.2
            )
            ax.add_patch(e)

    # Highlight the recovered injected asteroid in red
    if match_idx is not None:
        s = sources[match_idx]
        sx, sy = s['x'] - x0, s['y'] - y0
        from matplotlib.patches import Ellipse, Circle
        e = Ellipse(
            xy=(sx, sy),
            width  = 6 * s['a'],
            height = 6 * s['b'],
            angle  = np.degrees(s['theta']),
            edgecolor='red', facecolor='none', linewidth=2.5
        )
        ax.add_patch(e)
        ax.text(sx + 8, sy + 8, 'FAKE ASTEROID', color='red', fontsize=9,
                fontweight='bold')

    plt.tight_layout()
    plt.savefig('milestone2_detection.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: milestone2_detection.png")


# MAIN
if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 60)
    print("MILESTONE 2: Source Detection with SEP")
    print("=" * 60)

    # Step 1: Load and inject
    original, magzp, seeing, pixscale, gain = load_image(FITS_FILE)
    modified, stamp = inject_asteroid(
        original, magzp, seeing, pixscale, gain,
        mag=TARGET_MAG, x_pix=INJECT_X, y_pix=INJECT_Y
    )

    # Step 2: Detect sources in the injected image
    print(f"\nRunning SEP at {DETECT_THRESH}σ threshold...")
    sources, fwhm, bkg = detect_sources(modified, thresh=DETECT_THRESH)
    print(f"Found {len(sources)} sources total in the full image")
    print(f"Median FWHM of detected sources: {np.median(fwhm):.2f} pixels")
    print(f"  (Expected ~{seeing / pixscale:.2f} px from header SEEING={seeing}\")")

    # Step 3: Check if our fake asteroid was recovered
    x_int = int(round(INJECT_X))
    y_int = int(round(INJECT_Y))
    match_idx, match_dist = find_match(sources, INJECT_X, INJECT_Y, MATCH_RADIUS)

    print("\n--- RECOVERY CHECK ---")
    if match_idx is not None:
        s = sources[match_idx]
        measured_fwhm = 2.355 * (s['a'] + s['b']) / 2.0
        print(f"✓ RECOVERED! Distance from injected position: {match_dist:.2f} px")
        print(f"  Detected at : x={s['x']:.2f}, y={s['y']:.2f}")
        print(f"  Injected at : x={INJECT_X:.2f}, y={INJECT_Y:.2f}")
        print(f"  Detected flux : {s['flux']:.1f} ADU")
        print(f"  Detected FWHM : {measured_fwhm:.2f} px  (expected {seeing/pixscale:.2f} px)")
        print(f"  Peak SNR      : {s['peak'] / bkg.globalrms:.1f}σ")
    else:
        print(f"✗ NOT FOUND within {MATCH_RADIUS} px of injection site.")
        print(f"  Try lowering DETECT_THRESH (currently {DETECT_THRESH})")
        print(f"  Or try a brighter TARGET_MAG (currently {TARGET_MAG})")

    # Step 4: Plot
    data_sub = modified.astype(np.float32)
    data_sub = np.ascontiguousarray(data_sub)
    bkg2 = sep.Background(data_sub)
    plot_detection(data_sub - bkg2.back(), sources,
                   INJECT_X, INJECT_Y, match_idx)

    print("\nMilestone 2 complete.")
    print("Try TARGET_MAG=21 — does it still get detected? That's your limiting magnitude.")