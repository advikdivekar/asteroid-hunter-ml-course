from asthunt.viewer.list_viewer_units import list_viewer_units


def make_frame(folder, stamp, ccd=3, quadrant=4, products=("sciimg.fits", "mskimg.fits", "scimrefdiffimg.fits.fz", "diffimgpsf.fits")):
    for product in products:
        (folder / f"ztf_{stamp}_000618_zr_c{ccd:02d}_o_q{quadrant}_{product}").write_bytes(b"x")


def test_only_complete_nights_with_a_cached_table_are_listed(tmp_path):
    night = tmp_path / "ztf" / "2020-01-01"
    night.mkdir(parents=True)
    for k in range(5):
        make_frame(night, 20200101400000 + k)                                       # a full night
    for k in range(5):
        make_frame(night, 20200101500000 + k, quadrant=1)                           # no cached archive table for this quadrant
    for k in range(5):
        make_frame(night, 20200101600000 + k, ccd=12, quadrant=2, products=("sciimg.fits",))   # science image only: cannot run
    (tmp_path / "ztf" / "metadata__000618_c03_q4.csv").write_text("x")

    units = list_viewer_units(tmp_path, "zr", min_frames=5)

    assert units == [{"field": 618, "ccd": 3, "quadrant": 4, "night": "2020-01-01", "frames": 5}]


def test_a_night_with_too_few_frames_is_left_out(tmp_path):
    night = tmp_path / "ztf" / "2020-01-01"
    night.mkdir(parents=True)
    for k in range(3):
        make_frame(night, 20200101400000 + k)
    (tmp_path / "ztf" / "metadata__000618_c03_q4.csv").write_text("x")
    assert list_viewer_units(tmp_path, "zr", min_frames=5) == []
