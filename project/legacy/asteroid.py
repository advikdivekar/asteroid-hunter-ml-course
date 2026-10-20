from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astroquery.imcce import SkybotClass

Skybot = SkybotClass()

filename = '/Users/advikdivekar/Desktop/asteroid/ztf_test/sci/2018/0325/263518/ztf_20180325263518_000618_zg_c03_o_q4_sciimg.fits' 

hdul = fits.open(filename)
header = hdul[0].header
image = hdul[0].data
wcs = WCS(header) 

# Get image center and observation time from header
ra_center = header['CRVAL1']
dec_center = header['CRVAL2']
obs_jd = header['OBSJD']
 
center = SkyCoord(ra_center, dec_center, unit='deg')
obs_time = Time(obs_jd, format='jd')
 
print(f"Image center: RA={ra_center:.4f}°, Dec={dec_center:.4f}°")
print(f"Observation time: {obs_time.iso}")
print(f"Searching SkyBoT for known asteroids...\n")
 
# Query SkyBoT — search radius 0.6 deg to cover the full image
result = Skybot.cone_search(center, 0.6 * u.deg, obs_time)
 
print(f"FOUND {len(result)} KNOWN ASTEROIDS IN YOUR IMAGE!")
print("=" * 80)
print(f"{'Name':25s}  {'RA':>10s}  {'Dec':>10s}  {'Mag':>6s}  {'Pixel X':>8s}  {'Pixel Y':>8s}")
print("-" * 80)
 
# Convert each asteroid's sky position to pixel position
asteroid_pixels = []
for row in result:
    name = str(row['Name'])
    ra = row['RA'].value
    dec = row['DEC'].value
    mag = str(row['V'])
 
    # Sky → pixel
    sky_pos = SkyCoord(ra, dec, unit='deg')
    px, py = wcs.world_to_pixel(sky_pos)
 
    # Only keep if it's actually inside the image
    if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
        asteroid_pixels.append((name, ra, dec, mag, px, py))
        print(f"  {name:23s}  {ra:>10.5f}  {dec:>10.5f}  {mag:>6s}  {px:>8.1f}  {py:>8.1f}")
 
print(f"\n{len(asteroid_pixels)} asteroids are inside the image boundaries.")
 
# PLOT: Image with red circles on asteroids
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
 
vmin = np.nanpercentile(image, 5)
vmax = np.nanpercentile(image, 99)
 
# Left: full image with asteroid positions marked
ax = axes[0]
ax.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
for name, ra, dec, mag, px, py in asteroid_pixels:
    circle = plt.Circle((px, py), 20, color='red', fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.text(px + 25, py + 10, name, color='red', fontsize=6)
ax.set_title(f'All {len(asteroid_pixels)} Known Asteroids Marked', fontsize=13)
ax.set_xlabel('pixels')
ax.set_ylabel('pixels')
 
# Right: zoomed into the first asteroid found
if asteroid_pixels:
    name, ra, dec, mag, px, py = asteroid_pixels[0]
    size = 60  # pixels around the asteroid
    x0, x1 = int(max(0, px - size)), int(min(image.shape[1], px + size))
    y0, y1 = int(max(0, py - size)), int(min(image.shape[0], py + size))
    crop = image[y0:y1, x0:x1]
 
    ax = axes[1]
    ax.imshow(crop, cmap='gray', origin='lower',
              vmin=np.nanpercentile(crop, 5),
              vmax=np.nanpercentile(crop, 99))
    # Mark center
    cx, cy = px - x0, py - y0
    circle = plt.Circle((cx, cy), 15, color='red', fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.set_title(f'Zoomed: {name} (mag {mag})\nThis dot is a real asteroid!', fontsize=13)
    ax.set_xlabel('pixels')
 
plt.tight_layout()
plt.savefig('asteroids_found.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: asteroids_found.png")
print("\nIF THE RED CIRCLES LAND ON REAL DOTS → your coordinates are correct!")
print("This is the validation test (Steps 18-21). Run this every time something breaks.")
 
hdul.close()