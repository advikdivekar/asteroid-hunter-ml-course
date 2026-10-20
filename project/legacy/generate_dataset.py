import sys
import os
import glob
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
import sep
import galsim

warnings.filterwarnings('ignore')

FITS_FILES = sorted(glob.glob(
    '/Users/advikdivekar/Desktop/asteroid/ztf_test/sci/2018/*/*/*sciimg.fits'
))[:4]

DT_HOURS      = [0.0, 0.25, 0.50, 0.75]
DETECT_THRESH = 3.0
MATCH_RADIUS  = 5.0
LINEARITY_TOL = 3.0
MIN_RATE      = 0.5
MAX_RATE      = 60.0
MAGS          = np.arange(17.0, 21.5, 0.5)
RATES         = np.arange(5, 65, 10)
N_PER_COMBO   = 50
OUTPUT_CSV    = 'tracklets.csv'


def load_image(fits_path):
    hdul = fits.open(fits_path)
    data = hdul[0].data.astype(np.float64)
    hdr  = hdul[0].header
    hdul.close()
    return data, hdr['MAGZP'], hdr['SEEING'], hdr['PIXSCALE'], hdr['GAIN']


def inject_asteroid(data, magzp, seeing, pixscale, gain, mag, x_pix, y_pix, stamp_size=21):
    flux        = 10 ** (-(mag - magzp) / 2.5)
    fwhm_pixels = seeing / pixscale
    psf         = galsim.Moffat(beta=4.5, fwhm=fwhm_pixels)
    point       = galsim.DeltaFunction(flux=flux)
    source      = galsim.Convolve([point, psf])
    x_int = int(round(x_pix))
    y_int = int(round(y_pix))
    stamp = source.drawImage(
        nx=stamp_size, ny=stamp_size, scale=1.0,
        offset=galsim.PositionD(x_pix - x_int, y_pix - y_int)
    ).array.copy()
    stamp_e  = np.maximum(stamp * gain, 0)
    noisy    = np.random.poisson(stamp_e).astype(np.float64) / gain
    half     = stamp_size // 2
    modified = data.copy()
    modified[y_int-half:y_int+half+1, x_int-half:x_int+half+1] += noisy
    return modified


def detect_sources(data, thresh=3.0):
    data = np.ascontiguousarray(data.astype(np.float32))
    bkg  = sep.Background(data)
    return sep.extract(data - bkg.back(), thresh=thresh, err=bkg.rms())


def linearity_residual(positions):
    xs    = np.array([p[0] for p in positions])
    ys    = np.array([p[1] for p in positions])
    times = np.arange(len(positions), dtype=float)
    x_pred = np.polyval(np.polyfit(times, xs, 1), times)
    y_pred = np.polyval(np.polyfit(times, ys, 1), times)
    return np.sqrt((xs - x_pred)**2 + (ys - y_pred)**2).max()


def angular_rate(positions, dt_hours, pixscale):
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    ts = np.array(dt_hours)
    vx = np.polyfit(ts, xs, 1)[0]
    vy = np.polyfit(ts, ys, 1)[0]
    return np.sqrt(vx**2 + vy**2) * pixscale


def generate_positives(fits_files, dt_hours, mags, rates, n_per_combo):
    images, headers = [], []
    for f in fits_files:
        data, magzp, seeing, pixscale, gain = load_image(f)
        images.append(data)
        headers.append({'magzp': magzp, 'seeing': seeing,
                        'pixscale': pixscale, 'gain': gain})

    pixscale = headers[0]['pixscale']
    h, w     = images[0].shape
    pad      = int((rates.max() * max(dt_hours)) / pixscale + 50)

    tracklets, tracklet_id = [], 0

    print(f"Generating positives: {len(mags)*len(rates)*n_per_combo} injections")
    print(f"Image size: {w}x{h}  padding: {pad}px")

    for mag in tqdm(mags, desc="Magnitudes"):
        for rate in rates:
            for _ in range(n_per_combo):
                x0    = np.random.uniform(pad, w - pad)
                y0    = np.random.uniform(pad, h - pad)
                angle = np.random.uniform(0, 360)
                rad   = np.radians(angle)
                mx    = rate * np.cos(rad)
                my    = rate * np.sin(rad)

                true_pos = []
                for dt in dt_hours:
                    true_pos.append((
                        x0 + (mx / pixscale) * dt,
                        y0 + (my / pixscale) * dt
                    ))

                modified = []
                for i, img in enumerate(images):
                    hdr = headers[i]
                    modified.append(inject_asteroid(
                        img, hdr['magzp'], hdr['seeing'],
                        hdr['pixscale'], hdr['gain'],
                        mag=mag, x_pix=true_pos[i][0], y_pix=true_pos[i][1]
                    ))

                recovered, ok = [], True
                for i, mod in enumerate(modified):
                    srcs = detect_sources(mod, DETECT_THRESH)
                    tx, ty = true_pos[i]
                    dist = np.sqrt((srcs['x'] - tx)**2 + (srcs['y'] - ty)**2)
                    if len(dist) > 0 and dist.min() < MATCH_RADIUS:
                        idx = np.argmin(dist)
                        recovered.append((srcs[idx]['x'], srcs[idx]['y']))
                    else:
                        ok = False
                        break

                if not ok or len(recovered) < 4:
                    continue

                resid = linearity_residual(recovered)
                mrate = angular_rate(recovered, dt_hours, pixscale)

                if resid > LINEARITY_TOL or not (MIN_RATE < mrate < MAX_RATE):
                    continue

                tracklet_id += 1
                tracklets.append({
                    'tracklet_id'        : tracklet_id,
                    'mag'                : mag,
                    'true_rate'          : rate,
                    'true_angle'         : angle,
                    'measured_rate'      : mrate,
                    'linearity_residual' : resid,
                    'x1': recovered[0][0], 'y1': recovered[0][1],
                    'x2': recovered[1][0], 'y2': recovered[1][1],
                    'x3': recovered[2][0], 'y3': recovered[2][1],
                    'x4': recovered[3][0], 'y4': recovered[3][1],
                    'label': 1
                })

    print(f"Positive tracklets generated: {len(tracklets)}")
    return tracklets


def generate_bogus(fits_files, dt_hours, n_bogus):
    images = [load_image(f)[0] for f in fits_files]
    src_list = [
        np.array([(s['x'], s['y']) for s in detect_sources(img, DETECT_THRESH)])
        for img in images
    ]
    pixscale = load_image(fits_files[0])[3]

    bogus, bogus_id = [], 0

    for _ in tqdm(range(n_bogus), desc="Generating bogus"):
        positions, valid = [], True
        for srcs in src_list:
            if len(srcs) == 0:
                valid = False
                break
            positions.append(tuple(srcs[np.random.randint(0, len(srcs))]))
        if not valid:
            continue

        resid = linearity_residual(positions)
        mrate = angular_rate(positions, dt_hours, pixscale)

        if resid < 2.0 and MIN_RATE < mrate < MAX_RATE:
            continue

        bogus_id += 1
        bogus.append({
            'tracklet_id'        : bogus_id,
            'mag'                : np.nan,
            'true_rate'          : np.nan,
            'true_angle'         : np.nan,
            'measured_rate'      : mrate,
            'linearity_residual' : resid,
            'x1': positions[0][0], 'y1': positions[0][1],
            'x2': positions[1][0], 'y2': positions[1][1],
            'x3': positions[2][0], 'y3': positions[2][1],
            'x4': positions[3][0], 'y4': positions[3][1],
            'label': 0
        })

    print(f"Bogus tracklets generated: {len(bogus)}")
    return bogus


if __name__ == '__main__':
    np.random.seed(42)

    positives = generate_positives(FITS_FILES, DT_HOURS, MAGS, RATES, N_PER_COMBO)

    print(f"\nGenerating {len(positives)} bogus tracklets...")
    bogus = generate_bogus(FITS_FILES, DT_HOURS, len(positives))

    df = pd.DataFrame(positives + bogus).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Total rows : {len(df)}")
    print(f"Real        : {df['label'].sum()}")
    print(f"Bogus       : {len(df) - df['label'].sum()}")
    print(f"Columns     : {df.columns.tolist()}")