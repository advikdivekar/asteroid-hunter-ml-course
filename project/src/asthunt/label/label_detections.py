"""Give every detection a label: injected fake, known asteroid, or bogus."""

from asthunt.label.match_detections_to_positions import match_detections_to_positions


def label_detections(detections, injections, known_asteroids, match_radius, known_match_radius=None):
    """Return a copy of `detections` with three new columns.

    injection_id   : id of the matched fake asteroid, or -1
    known_asteroid : name of the matched known asteroid, or ""
    label          : 1 = real (a fake we planted, or a known asteroid)
                     0 = bogus (everything else)

    `injections` and `known_asteroids` must already be for THIS frame
    (columns x, y; injections also needs injection_id).

    known_match_radius: radius for known asteroids (defaults to match_radius).
    Catalogue predictions are less exact than our own fake positions:
    SkyBoT's are systematically ~2.9 arcsec off for ZTF, so we use 4 px.

    Honest caveat: "bogus" also contains the rare real object that no
    catalogue knows yet. That's label noise we accept (and ultimately the
    thing we are searching for).
    """
    labelled = detections.copy()

    injection_index = match_detections_to_positions(
        labelled["x"], labelled["y"], injections["x"], injections["y"], match_radius)
    labelled["injection_id"] = -1
    found = injection_index >= 0
    if found.any():
        labelled.loc[found, "injection_id"] = injections["injection_id"].to_numpy()[injection_index[found]]

    known_index = match_detections_to_positions(
        labelled["x"], labelled["y"], known_asteroids["x"], known_asteroids["y"], known_match_radius or match_radius)
    labelled["known_asteroid"] = ""
    known = known_index >= 0
    if known.any():
        labelled.loc[known, "known_asteroid"] = known_asteroids["name"].to_numpy()[known_index[known]]

    labelled["label"] = (found | known).astype(int)
    return labelled
