from asthunt.deep.group_frames_into_stacks import group_frames_into_stacks


def test_groups_by_size_and_breaks_at_gaps():
    minute = 1 / (24 * 60)
    times = [0, 30, 34, 38, 42, 46, 50, 54, 58, 62]          # one early frame, then a 4-minute cadence
    stacks = group_frames_into_stacks([t * minute for t in times], frames_per_stack=4, max_gap_minutes=10)
    assert stacks == [[1, 2, 3, 4], [5, 6, 7, 8]]            # the early frame and the short leftover are dropped


def test_a_night_needs_20_frames_for_the_three_stacks_the_linker_requires():
    # configs/deep.yaml: stacks of 8, and a tracklet needs >= 3 stacks. 19 frames leave a 3-frame
    # leftover (dropped), so only 2 stacks: the deep search could never find anything on such a night.
    minute = 1 / (24 * 60)
    stacks = lambda n: group_frames_into_stacks([i * 4 * minute for i in range(n)], frames_per_stack=8, max_gap_minutes=15)
    assert len(stacks(19)) == 2
    assert len(stacks(20)) == 3
