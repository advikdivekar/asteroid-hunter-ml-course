"""Write an HTML page showing each candidate tracklet for a human to vet."""

import html

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")       # draw to files, no window


def write_candidate_report(candidates, members, detections, stamps, out_dir, title):
    """Write out_dir/index.html plus one PNG per candidate; return the HTML path.

    candidates : tracklets to show (ordered: most promising first), with
                 tracklet_id, n_detections, rate_arcsec_per_hour, tracklet_score
    stamps     : normalized stamps for every row of `detections` (same order)

    Each candidate gets a picture: one column per detection, rows =
    science / reference / difference. A real asteroid shows a moving dot in
    science and difference, and nothing in reference.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    row_of = {detection_id: position for position, detection_id in enumerate(detections.index)}
    blocks = []

    for _, candidate in candidates.iterrows():
        ids = members.loc[members["tracklet_id"] == candidate["tracklet_id"], "detection_id"].tolist()
        rows = detections.loc[ids].sort_values("obsjd")

        figure, axes = plt.subplots(3, len(rows), figsize=(1.8 * len(rows), 5.4), squeeze=False)
        for column, (detection_id, detection) in enumerate(rows.iterrows()):
            for channel, name in enumerate(["sci", "ref", "diff"]):
                axes[channel, column].imshow(stamps[row_of[detection_id], channel], origin="lower",
                                             cmap="gray", vmin=-3, vmax=10)
                axes[channel, column].set_xticks([]); axes[channel, column].set_yticks([])
                if column == 0:
                    axes[channel, column].set_ylabel(name)
            axes[0, column].set_title(f"frame {detection['frame_index']}\nSNR {detection['snr']:.1f}", fontsize=8)
        image_name = f"tracklet_{int(candidate['tracklet_id'])}.png"
        figure.savefig(out_dir / image_name, dpi=80, bbox_inches="tight")
        plt.close(figure)

        first = rows.iloc[0]
        blocks.append(
            f"<h3>Tracklet {int(candidate['tracklet_id'])}</h3>"
            f"<p>score {candidate['tracklet_score']:.3f} &middot; {int(candidate['n_detections'])} detections &middot; "
            f"{candidate['rate_arcsec_per_hour']:.1f} \"/hr &middot; mag {rows['mag'].mean():.2f} &middot; "
            f"RA {first['ra']:.5f}, Dec {first['dec']:.5f}</p>"
            f"<img src=\"{image_name}\">"
        )

    page = (f"<!doctype html><html><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title>"
            f"<style>body{{font-family:sans-serif;max-width:1000px;margin:auto;padding:16px}}img{{max-width:100%}}</style>"
            f"</head><body><h1>{html.escape(title)}</h1><p>{len(candidates)} candidates, most promising first.</p>"
            + "\n".join(blocks) + "</body></html>")
    path = out_dir / "index.html"
    path.write_text(page)
    return path
