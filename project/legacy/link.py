import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import sep

sys.path.insert(0, '/Users/advikdivekar/Desktop/asteroid')
from inject import load_image, inject_asteroid

FITS_FILES = sorted(glob.glob(
    '/Users/advikdivekar/Desktop/asteroid/ztf_test/sci/2018/*/*/*sciimg.fits'
))[:4]

TARGET_MAG       = 19.0
START_X          = 1500.0
START_Y          = 1500.0
MOTION_RA        = 20.0
MOTION_DEC       = 5.0
DETECT_THRESH    = 3.0
MATCH_RADIUS     = 5.0
MAX_RATE_ARCSEC  = 60.0
MIN_RATE_ARCSEC  = 0.5
LINEARITY_TOL    = 2.0
SIMULATED_DT_HOURS = [0.0, 0.25, 0.50, 0.75]


def get_obs_time(fits_path):
    hdul = fits.open(fits_path)
    jd = hdul[0].header['OBSJD']
    hdul.close()
    return jd


def compute_injection_position(start_x, start_y, motion_x_arcsec_hr,
                                motion_y_arcsec_hr, pixscale, dt_hours):
    motion_x_pix_hr = motion_x_arcsec_hr / pixscale
    motion_y_pix_hr = motion_y_arcsec_hr / pixscale
    x = start_x + motion_x_pix_hr * dt_hours
    y = start_y + motion_y_pix_hr * dt_hours
    return x, y


def detect_sources_in_frame(fits_path, injected_image, thresh=3.0):
    data = injected_image.astype(np.float32)
    data = np.ascontiguousarray(data)
    bkg = sep.Background(data)
    data_sub = data - bkg.back()
    sources = sep.extract(data_sub, thresh=thresh, err=bkg.rms())
    return sources, bkg.globalrms


def linearity_residual(positions):
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    times = np.arange(len(positions), dtype=float)
    px = np.polyfit(times, xs, 1)
    py = np.polyfit(times, ys, 1)
    x_pred = np.polyval(px, times)
    y_pred = np.polyval(py, times)
    residuals = np.sqrt((xs - x_pred)**2 + (ys - y_pred)**2)
    return residuals.max()


def angular_rate_arcsec_hr(positions, dt_hours_list, pixscale):
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    ts = np.array(dt_hours_list)
    vx_pix = np.polyfit(ts, xs, 1)[0]
    vy_pix = np.polyfit(ts, ys, 1)[0]
    speed_pix_hr = np.sqrt(vx_pix**2 + vy_pix**2)
    return speed_pix_hr * pixscale


if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 60)
    print("MILESTONE 3: Four-Frame Injection and Tracklet Linking")
    print("=" * 60)

    if len(FITS_FILES) < 4:
        print(f"ERROR: Only found {len(FITS_FILES)} FITS files. Need at least 4.")
        sys.exit(1)

    FITS_FILES = FITS_FILES[:4]
    print(f"\nUsing these 4 files:")
    for f in FITS_FILES:
        print(f"  {os.path.basename(f)}")

    dt_hours = SIMULATED_DT_HOURS
    print(f"\nSimulated time offsets (hours):")
    for i, dt in enumerate(dt_hours):
        print(f"  Frame {i}: dt = {dt:.3f} hr")

    _, magzp0, seeing0, pixscale, gain0 = load_image(FITS_FILES[0])

    true_positions = []
    for i, dt in enumerate(dt_hours):
        x, y = compute_injection_position(
            START_X, START_Y, MOTION_RA, MOTION_DEC, pixscale, dt
        )
        true_positions.append((x, y))
        print(f"  Frame {i}: inject at ({x:.2f}, {y:.2f})")

    total_motion_x = true_positions[-1][0] - true_positions[0][0]
    total_motion_y = true_positions[-1][1] - true_positions[0][1]
    total_motion_px = np.sqrt(total_motion_x**2 + total_motion_y**2)
    print(f"\nTotal asteroid motion across 4 frames: {total_motion_px:.1f} pixels")

    all_sources = []
    all_rms = []

    for i, fits_path in enumerate(FITS_FILES):
        original, magzp, seeing, pixscale, gain = load_image(fits_path)
        x_inj, y_inj = true_positions[i]
        modified, _ = inject_asteroid(
            original, magzp, seeing, pixscale, gain,
            mag=TARGET_MAG, x_pix=x_inj, y_pix=y_inj
        )
        sources, rms = detect_sources_in_frame(fits_path, modified, DETECT_THRESH)
        all_sources.append(sources)
        all_rms.append(rms)
        print(f"  Frame {i}: {len(sources)} sources detected, RMS = {rms:.2f} ADU")

    print("\nInjection recovery per frame:")
    recovered_positions = []
    all_recovered = True

    for i, (sources, (x_true, y_true)) in enumerate(zip(all_sources, true_positions)):
        dist = np.sqrt((sources['x'] - x_true)**2 + (sources['y'] - y_true)**2)
        idx = np.argmin(dist)
        if dist[idx] < MATCH_RADIUS:
            rx, ry = sources[idx]['x'], sources[idx]['y']
            recovered_positions.append((rx, ry))
            print(f"  Frame {i}: recovered at ({rx:.2f}, {ry:.2f}), offset {dist[idx]:.2f} px")
        else:
            print(f"  Frame {i}: NOT recovered (closest: {dist[idx]:.2f} px away)")
            recovered_positions.append(None)
            all_recovered = False

    print("\nTracklet linking:")
    valid_positions = [p for p in recovered_positions if p is not None]
    valid_times = [dt_hours[i] for i, p in enumerate(recovered_positions) if p is not None]

    if len(valid_positions) < 2:
        print("Not enough detections to link.")
    else:
        rate = angular_rate_arcsec_hr(valid_positions, valid_times, pixscale)
        resid = linearity_residual(valid_positions)

        expected_rate = np.sqrt(MOTION_RA**2 + MOTION_DEC**2)
        print(f"  Motion rate   : {rate:.2f} arcsec/hr  (expected {expected_rate:.2f})")
        print(f"  Linearity RMS : {resid:.3f} px  (tolerance: {LINEARITY_TOL} px)")

        rate_ok   = MIN_RATE_ARCSEC < rate < MAX_RATE_ARCSEC
        linear_ok = resid < LINEARITY_TOL
        frames_ok = len(valid_positions) == 4

        print(f"\n  Rate in main-belt range : {'YES' if rate_ok else 'NO'}")
        print(f"  Linear motion           : {'YES' if linear_ok else 'NO'}")
        print(f"  All 4 frames recovered  : {'YES' if frames_ok else f'NO ({len(valid_positions)}/4)'}")

        if rate_ok and linear_ok and frames_ok:
            print("\nVALID TRACKLET FOUND")
        else:
            print("\nTracklet rejected.")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Four-frame injection  mag={TARGET_MAG}  rate={np.sqrt(MOTION_RA**2+MOTION_DEC**2):.0f} arcsec/hr')

    window = 80
    for i, (fits_path, (x_true, y_true)) in enumerate(zip(FITS_FILES, true_positions)):
        original, magzp, seeing, pixscale, gain = load_image(fits_path)
        modified, _ = inject_asteroid(
            original, magzp, seeing, pixscale, gain,
            mag=TARGET_MAG, x_pix=x_true, y_pix=y_true
        )
        half = window // 2
        x_int, y_int = int(round(x_true)), int(round(y_true))
        crop = modified[y_int-half:y_int+half, x_int-half:x_int+half]
        ax = axes[i]
        ax.imshow(crop, cmap='gray', origin='lower',
                  vmin=np.nanpercentile(crop, 5),
                  vmax=np.nanpercentile(crop, 99))
        ax.axhline(half, color='red', lw=0.7, alpha=0.6)
        ax.axvline(half, color='red', lw=0.7, alpha=0.6)
        ax.set_title(f'Frame {i}  dt={dt_hours[i]:.2f}hr\n({x_true:.1f}, {y_true:.1f})')
        ax.set_xlabel('pixels')

    plt.tight_layout()
    plt.savefig('milestone3_tracklet.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: milestone3_tracklet.png")
    print("Milestone 3 complete.")