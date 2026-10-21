import numpy as np
import soundfile as sf

from valorant_voice import config
from valorant_voice.data import audio, ingest, manifest, split


def tone(seconds, rate=22050, amplitude=0.3, frequency=220.0):
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype("float32")


def padded(clip, rate=22050, lead=0.5, tail=0.5):
    silence = lambda seconds: np.zeros(int(seconds * rate), dtype="float32")
    return np.concatenate([silence(lead), clip, silence(tail)])


def test_trim_silence_removes_padding_but_keeps_the_tone():
    clip = padded(tone(2.0))
    trimmed = audio.trim_silence(clip, 22050)
    assert 2.0 <= audio.duration(trimmed, 22050) < 2.2


def test_trim_silence_of_pure_silence_is_empty():
    assert len(audio.trim_silence(np.zeros(22050, dtype="float32"), 22050)) == 0


def test_normalise_loudness_hits_target_and_never_clips():
    quiet = audio.normalise_loudness(tone(1.0, amplitude=0.01))
    rms_db = 20 * np.log10(np.sqrt((quiet**2).mean()))
    assert abs(rms_db - audio.TARGET_RMS_DBFS) < 0.5
    loud = audio.normalise_loudness(tone(1.0, amplitude=0.99) * 1.0)
    assert np.abs(loud).max() <= 0.97 + 1e-6


def test_resample_changes_length_by_the_rate_ratio():
    out = audio.resample(tone(1.0, rate=48000), 48000, 22050)
    assert abs(len(out) - 22050) <= 1


def test_clean_clip_drops_short_and_long_clips():
    clip, reason = ingest.clean_clip(padded(tone(0.5)), 22050)
    assert clip is None and reason == "too_short"
    clip, reason = ingest.clean_clip(tone(config.MAX_SECONDS + 3), 22050)
    assert clip is None and reason == "too_long"
    clip, reason = ingest.clean_clip(padded(tone(3.0)), 22050)
    assert clip is not None and reason is None


def test_ingest_agent_writes_clean_clips_and_counts_outcomes(tmp_path):
    raw, clean = tmp_path / "raw", tmp_path / "clean"
    (raw / "sage").mkdir(parents=True)
    sf.write(raw / "sage" / "good.wav", padded(tone(3.0, rate=44100), rate=44100), 44100)
    sf.write(raw / "sage" / "short.wav", tone(0.3), 22050)
    (raw / "sage" / "notes.txt").write_text("not audio")
    (raw / "sage" / "broken.wav").write_bytes(b"not a wav")

    outcome = ingest.ingest_agent("sage", raw_dir=raw, clean_dir=clean)

    assert outcome == {"kept": 1, "too_short": 1, "unreadable": 1}
    written, rate = sf.read(clean / "sage" / "good.wav")
    assert rate == config.SAMPLE_RATE and written.ndim == 1


def test_split_is_deterministic_and_roughly_proportional():
    ids = [f"sage/line_{i}" for i in range(4000)]
    first = [split.assign_split(i, seed=1) for i in ids]
    assert first == [split.assign_split(i, seed=1) for i in ids]
    test_share = first.count("test") / len(ids)
    val_share = first.count("val") / len(ids)
    assert abs(test_share - config.TEST_FRACTION) < 0.02
    assert abs(val_share - config.VAL_FRACTION) < 0.02


def test_split_does_not_move_a_clip_when_others_are_added():
    before = split.assign_split("jett/line_7")
    _ = [split.assign_split(f"jett/line_{i}") for i in range(100)]
    assert split.assign_split("jett/line_7") == before


def test_build_manifests_never_puts_a_clip_in_two_splits(tmp_path):
    clean, out = tmp_path / "clean", tmp_path / "manifests"
    for agent in ("sage", "jett"):
        (clean / agent).mkdir(parents=True)
        lines = [(f"line_{i}", f"Line number {i}") for i in range(300)]
        (clean / agent / "transcripts.csv").write_text("\n".join(f"{a}|{b}" for a, b in lines), encoding="utf-8")

    counts = manifest.build_manifests(clean_dir=clean, out_dir=out)

    assert sum(counts.values()) == 600
    seen = []
    for name in ("train", "val", "test"):
        rows = (out / f"{name}.csv").read_text(encoding="utf-8").splitlines()
        assert rows[0] == "audio_file|text|speaker_name"
        seen += [row.split("|")[0] for row in rows[1:]]
    assert len(seen) == len(set(seen)) == 600
