from asthunt.label import refine_missing_known_positions as module


def test_only_exposures_without_horizons_positions_are_refined(tmp_path, monkeypatch):
    folder = tmp_path / "known_asteroids"
    folder.mkdir()
    (folder / "known_asteroids_horizons__done.csv").write_text("name,ra,dec,v_mag,x,y\n")
    asked = []
    monkeypatch.setattr(module, "refine_known_positions",
                        lambda paths, interim, max_v_mag: asked.extend(p.name for p in paths) or 3)

    count = module.refine_missing_known_positions([tmp_path / "done_sciimg.fits", tmp_path / "new_sciimg.fits"], tmp_path)

    assert asked == ["new_sciimg.fits"]
    assert count == 3


def test_nothing_is_asked_when_every_exposure_is_refined(tmp_path, monkeypatch):
    folder = tmp_path / "known_asteroids"
    folder.mkdir()
    (folder / "known_asteroids_horizons__done.csv").write_text("name,ra,dec,v_mag,x,y\n")
    monkeypatch.setattr(module, "refine_known_positions", lambda *a, **k: (_ for _ in ()).throw(AssertionError("asked")))

    assert module.refine_missing_known_positions([tmp_path / "done_sciimg.fits"], tmp_path) == 0
