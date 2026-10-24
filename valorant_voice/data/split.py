"""Deterministic train, validation and test assignment.

A clip's split depends only on its id and the seed, so adding clips later never moves an
existing clip between splits and the test lines are never trained on.
"""

import hashlib

from .. import config


def assign_split(clip_id: str, seed: int = config.SPLIT_SEED) -> str:
    digest = hashlib.sha1(f"{seed}:{clip_id}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < config.TEST_FRACTION:
        return "test"
    if position < config.TEST_FRACTION + config.VAL_FRACTION:
        return "val"
    return "train"
