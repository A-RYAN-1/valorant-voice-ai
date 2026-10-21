from valorant_voice.tts.references import choose_references


def test_choose_references_prefers_clips_near_nine_seconds():
    clips = [("a", 3.0), ("b", 8.5), ("c", 9.2), ("d", 11.9), ("e", 6.5), ("f", 30.0)]
    assert choose_references(clips, count=3) == ["c", "b", "e"]


def test_choose_references_returns_nothing_when_no_clip_fits():
    assert choose_references([("a", 2.0), ("b", 20.0)]) == []
