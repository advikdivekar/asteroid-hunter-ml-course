# 02. Domain primer: the astronomy you need (no prior astronomy assumed)

You do not need to be an astronomer. You need about twelve ideas, each explained so you can say it in your own words in a viva.

## 1. What an asteroid is, and why it shows up as a moving dot
Asteroids are rocky bodies orbiting the Sun (most between Mars and Jupiter, the *main belt*). From Earth, stars are so far away that they
appear fixed. An asteroid is close enough that it visibly **moves against the stars**: tens of arcseconds per hour for a typical main-belt asteroid
near opposition (our data: 28-48 arcsec per hour). An arcsecond is 1/3600 of a degree.
Over the ~30 minutes of 5 consecutive exposures, an asteroid moves in a **nearly straight line at a nearly constant speed**. That is the physical fact the whole *linking* stage relies on.

## 2. The telescope and the survey
**ZTF (Zwicky Transient Facility)** photographs the northern sky with a camera at Palomar Observatory (California). The camera has 16 CCD chips; each chip is read out in
4 **quadrants** of about 3072 x 3080 pixels. One pixel is about **1.01 arcseconds**. A standard exposure lasts **30 seconds**. Images come in filters (we use **r band**, "zr").
The data is public through **IRSA** (a NASA/IPAC archive).

A **unit** in our project = one *field* (a patch of sky) + one *CCD* + one *quadrant* + one *night*. Example: `000618_c03_q4_2019-02-08` = field 618, CCD 3, quadrant 4, night 8 Feb 2019.

## 3. The four files per exposure
| File | What it is |
|---|---|
| science image | the picture itself (about 38 MB) |
| reference image | a deep picture of the same sky built from many old exposures (we recompute it as science minus difference) |
| **difference image** | science minus the (blurred, aligned) reference: static things cancel |
| mask | 16 yes/no flags per pixel: saturated, dead, cosmic ray, ghost, ... |
| PSF | the blur shape of a point source in the difference image |

## 4. Difference imaging: the key trick
`difference = science - reference`. Stars, galaxies and the sky glow are the same in both and cancel. What remains:
new or moving things (**positive blobs**, what we want), faded things (negative blobs, ignored), imperfect cancellation of bright stars (**dipoles**), cosmic rays, and noise.
Most blobs in a difference image are junk, which is exactly why a classifier is needed.

## 5. Noise, sigma, and signal-to-noise (SNR)
Every pixel has random noise. We measure its spread (**sigma**) robustly (with the *median absolute deviation*, MAD, so bright blobs do not fool it:
sigma = 1.4826 x median(|x - median|)). A blob "5 sigma" bright is 5 times the noise level. **SNR** = measured flux divided by its uncertainty.
With pure noise, about 1 pixel in 3.5 million exceeds +5 sigma, so a 5-sigma blob is unlikely to be chance, *unless you looked in many places* (see 12).

## 6. The PSF (point spread function)
A star or asteroid is a point of light, but the atmosphere and optics blur it into a small round blob (about 2 pixels wide for us). The blur shape is the **PSF**.
Knowing it lets us do **matched filtering**: smooth the image with the blur shape, which makes faint point-like sources stand out of random noise.

## 7. Magnitudes (how astronomers measure brightness)
A **magnitude** is a log brightness scale where **bigger = fainter**. 5 magnitudes = a factor of 100 in brightness. So one magnitude is a factor of about 2.5.
Our single-frame limit is about **magnitude 20.5-21.4**; the deep search reaches about **22.1**, which is 0.7 magnitudes fainter, a factor of about 1.9 in brightness.
The faintest star the naked eye can see is about magnitude 6, so these asteroids are about a million times fainter (15 magnitudes = a factor of 10^6).

## 7b. Opposition
At **opposition** the asteroid is opposite the Sun in the sky: closest to Earth and brightest, and there are the most of them near the *ecliptic* (the plane of the planets).
Our test night, 2019-02-08, was near opposition, so it is both the most asteroid-rich and a deliberate "distribution shift" from the training nights.

## 8. WCS (sky coordinates) and why pointing matters
The **WCS** converts pixel positions to sky positions (RA, Dec). The telescope does not point at exactly the same pixels in consecutive exposures (shifts up to 12 pixels). An asteroid moves in a straight
line **on the sky**, not in each frame's pixels. So we put every blob on frame 1's common sky grid before linking. Forgetting this was our biggest bug.

## 9. Known versus unknown asteroids
Hundreds of thousands of asteroids are catalogued. The **Minor Planet Center (MPC)** keeps the official list. We look them up with **SkyBoT** (a service that predicts where known objects are)
and **JPL Horizons** (NASA's very accurate ephemeris). We measured SkyBoT positions to be about 2.9 arcsec (2.7 px) off, while Horizons is within 0.26 px. Almost everything we detect is **already known**; only objects matching nothing are *candidates*.

## 10. Tracklet
A **tracklet** = a set of detections in different frames that line up as one moving object. It needs at least 3 of the 5 frames, a straight line within 1.5 pixels, and a speed between 8 and 120 arcsec/hour.

## 11. Injection-recovery and completeness
We cannot know which real asteroids are in an image, so we **plant fake asteroids** (known brightness, speed, position) into the real images, run the normal pipeline and count how many come out.
The fraction recovered, as a function of brightness, is the **completeness**. "50% completeness at magnitude 20.5" means: half of the fakes of that brightness are found. This is the standard method in survey astronomy.

## 12. Look-elsewhere effect (why we need a 6-sigma cut in the deep search)
If you test thousands of places or motions, some random noise will reach 5 sigma by chance. In the deep search we try 4,888 motions at every pixel, so noise alone produced 6,179 peaks at 5 sigma but only 44 at 6 sigma (on a pure-noise test). So we require 6 sigma. This is the same logic as multiple-testing correction in statistics.

## Two more terms you will hear
- **Dipole**: a bright half next to a dark half, the typical leftover of a bright star that did not subtract perfectly.
- **Ghost / bleed trail**: optical or detector artifacts (a bright star's charge spilling along a CCD column). Bleed trails line up along columns and can fool a fast-object search; we found six of seven fast "tracks" we inspected were such artifacts.

## Self-test (say each in one sentence)
1. Why do we use difference images? 2. What does "5 sigma" mean? 3. Which is brighter, magnitude 18 or 21? By how much? 4. What is opposition and why did we choose such a night for the test?
5. Why must pointing shifts be corrected before linking? 6. What is a tracklet? 7. What does 50% completeness mean? 8. Why can an astronomer not just trust one 5-sigma blob?
