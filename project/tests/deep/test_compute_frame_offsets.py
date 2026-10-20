from astropy.wcs import WCS

from asthunt.deep.compute_frame_offsets import compute_frame_offsets


def wcs_with_shift(shift):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [135.6, 22.8]
    wcs.wcs.crpix = [100.0 + shift[0], 100.0 + shift[1]]
    wcs.wcs.cd = [[-1 / 3600, 0], [0, 1 / 3600]]
    return wcs


def test_offsets_match_the_pointing_shifts():
    offsets = compute_frame_offsets([wcs_with_shift((0, 0)), wcs_with_shift((7, -3)), wcs_with_shift((-12, 5))], (200, 200))
    assert offsets.tolist() == [[0, 0], [7, -3], [-12, 5]]
