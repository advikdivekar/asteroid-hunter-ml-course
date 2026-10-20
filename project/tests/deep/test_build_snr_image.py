import numpy as np

from asthunt.deep.build_snr_image import build_snr_image


def gaussian_psf(size=15, sigma=1.0):
    yy, xx = np.indices((size, size)) - size // 2
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def test_noise_becomes_unit_normal_and_a_source_stands_out():
    rng = np.random.default_rng(0)
    image = rng.normal(0, 7.0, size=(200, 200))                    # any noise level...
    psf = gaussian_psf()
    image[100 - 7:100 + 8, 60 - 7:60 + 8] += 400 * psf             # ...plus one point source
    bad = np.zeros(image.shape, dtype=bool)
    bad[0:10, :] = True

    snr = build_snr_image(image, bad, psf)

    noise = snr[20:80, 120:190]
    assert abs(noise.std() - 1.0) < 0.1                            # pure noise -> std 1
    assert snr[100, 60] > 10                                       # the source is many sigma
    assert (snr[0:10, :] == 0).all()                               # bad pixels -> 0
