"""Split a night's frames into groups of consecutive exposures to stack together."""


def group_frames_into_stacks(frame_times_jd, frames_per_stack, max_gap_minutes):
    """Return a list of stacks, each a list of frame positions (indices into frame_times_jd, time order).

    A new stack starts after `frames_per_stack` frames, or when the time gap to
    the previous frame exceeds `max_gap_minutes` (e.g. the telescope went
    elsewhere for a while). Stacks shorter than half the target size are dropped:
    too few frames to gain much depth.
    """
    order = sorted(range(len(frame_times_jd)), key=lambda i: frame_times_jd[i])
    stacks, current = [], []
    for i in order:
        if current:
            gap = (frame_times_jd[i] - frame_times_jd[current[-1]]) * 24 * 60
            if len(current) == frames_per_stack or gap > max_gap_minutes:
                stacks.append(current)
                current = []
        current.append(i)
    if current:
        stacks.append(current)
    return [stack for stack in stacks if len(stack) >= frames_per_stack / 2]
