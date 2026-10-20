import pytest

from asthunt.cli import main


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["fly"])


def test_search_requires_a_night():
    with pytest.raises(SystemExit):
        main(["search", "--field", "618", "--ccd", "3", "--quadrant", "4"])
