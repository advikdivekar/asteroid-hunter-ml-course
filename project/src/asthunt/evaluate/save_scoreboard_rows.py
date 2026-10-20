"""Keep reports/scoreboard.csv: one row per (version, split)."""

import pandas as pd


def save_scoreboard_rows(rows, scoreboard_file):
    """Add rows to the scoreboard file and return the full table.

    Each row must have "version" and "split". If the file already holds a row
    for the same (version, split), it is replaced, so re-running an evaluation
    never creates duplicates, and older versions' rows stay untouched.
    """
    new = pd.DataFrame(rows)
    if scoreboard_file.exists():
        old = pd.read_csv(scoreboard_file)
        keys = set(zip(new["version"], new["split"]))
        old = old[[(v, s) not in keys for v, s in zip(old["version"], old["split"])]]
        new = pd.concat([old, new], ignore_index=True)

    new = new.sort_values(["split", "version"]).reset_index(drop=True)
    scoreboard_file.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(scoreboard_file, index=False)
    return new
