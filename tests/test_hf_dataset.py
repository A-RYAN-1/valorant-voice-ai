import io

import numpy as np
import pytest
import soundfile as sf

from valorant_voice.data import fetch, manifest, prepare, stats, text
from valorant_voice.data.transcribe import matches_audio


def wav_bytes(seconds=3.0, rate=48000):
    buffer = io.BytesIO()
    t = np.arange(int(seconds * rate)) / rate
    sf.write(buffer, (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), rate, format="WAV")
    return buffer.getvalue()


def row(name, transcript, audio=True):
    return {
        "audio_name": name,
        "transcript": transcript,
        "audio_file": {"bytes": wav_bytes(), "path": f"{name}.wav"} if audio else None,
    }


def test_clean_text_strips_the_wrapping_quotes_only():
    assert text.clean_text('  "Must I do   everything?" ') == "Must I do everything?"
    assert text.clean_text("It's fine") == "It's fine"
    assert text.clean_text("“Okay.”") == "Okay."


def test_text_key_ignores_case_and_punctuation():
    assert text.text_key("Okay.") == text.text_key("okay!") == "okay"
    assert text.text_key("...") == ""


def test_write_rows_saves_audio_and_cleaned_text_and_skips_rows_without_audio(tmp_path):
    rows = [row("SageAce1", '"Must I do everything?"'), row("Bad/Name 2", '"Hi."'), row("NoAudio", '"x"', audio=False)]

    assert fetch.write_rows("sage", rows, raw_dir=tmp_path) == 2

    assert (tmp_path / "sage" / "SageAce1.wav").exists()
    assert (tmp_path / "sage" / "Bad_Name_2.wav").exists()
    lines = (tmp_path / "sage" / fetch.SOURCE_TRANSCRIPTS).read_text(encoding="utf-8").splitlines()
    assert lines == ["SageAce1|Must I do everything?", "Bad_Name_2|Hi."]


def test_fetch_refuses_an_agent_the_dataset_lacks():
    with pytest.raises(ValueError):
        fetch.fetch_agent("waylay")


def test_prepare_keeps_one_clip_per_text_and_only_clips_that_survived_ingest(tmp_path):
    raw, clean = tmp_path / "raw", tmp_path / "clean"
    fetch.write_rows("sage", [row("a", '"Okay."'), row("b", '"okay!"'), row("c", '"Hello there."'),
                              row("d", '"..."'), row("e", '"Lost in ingest."')], raw_dir=raw)
    (clean / "sage").mkdir(parents=True)
    for clip_id in "abcd":
        (clean / "sage" / f"{clip_id}.wav").write_bytes(b"")

    outcome = prepare.prepare_agent("sage", raw_dir=raw, clean_dir=clean)

    assert outcome == {"kept": 2, "duplicate": 1, "no_text": 1}
    assert (clean / "sage" / "transcripts.csv").read_text(encoding="utf-8").splitlines() == ["a|Okay.", "c|Hello there."]


def test_the_same_text_never_lands_in_two_splits_across_agents(tmp_path):
    clean = tmp_path / "clean"
    for agent in ("sage", "jett", "omen"):
        (clean / agent).mkdir(parents=True)
        rows = [f"{agent}_{i}|Shared line number {i % 150}" for i in range(300)]
        (clean / agent / "transcripts.csv").write_text("\n".join(rows), encoding="utf-8")

    manifest.build_manifests(clean_dir=clean, out_dir=tmp_path / "out")

    split_of = {}
    for name in ("train", "val", "test"):
        for line in (tmp_path / "out" / f"{name}.csv").read_text(encoding="utf-8").splitlines()[1:]:
            line_text = line.split("|")[1]
            assert split_of.setdefault(text.text_key(line_text), name) == name


def test_matches_audio_allows_small_differences_only():
    assert matches_audio("Must I do everything?", "must I do everything")
    assert matches_audio("Today, I was the sword!", "Today I was the sword")
    assert not matches_audio("Must I do everything?", "thank you for watching")


def test_audit_counts_minutes_drops_and_splits(tmp_path):
    folder = tmp_path / "sage"
    folder.mkdir()
    sf.write(folder / "a.wav", np.zeros(22050 * 30, dtype="float32"), 22050)
    sf.write(folder / "b.wav", np.zeros(22050 * 30, dtype="float32"), 22050)
    (folder / "transcripts.csv").write_text("a|One\nb|Two", encoding="utf-8")
    stats.record(folder, "fetch", {"fetched": 5})
    stats.record(folder, "ingest", {"kept": 3, "too_short": 1, "unreadable": 1})
    stats.record(folder, "prepare", {"kept": 2, "duplicate": 1})

    row_ = stats.audit_agent(folder)

    assert row_["fetched"] == 5 and row_["kept"] == 2 and row_["minutes"] == 1.0
    assert row_["too_short"] == 1 and row_["unreadable"] == 1 and row_["duplicate"] == 1
    assert row_["train"] + row_["val"] + row_["test"] == 2
    table = stats.audit_table(tmp_path)
    assert "| sage | 5 | 2 | 1.0 |" in table and "**total**" in table


def test_length_bucket_names_the_range_a_clip_falls_in():
    from valorant_voice.data.ingest import length_bucket

    assert length_bucket(0.3) == "len_under_0.5s"
    assert length_bucket(1.2) == "len_under_1.5s"
    assert length_bucket(30) == "len_over_12s"


def test_whisper_input_is_mono_16k_float(tmp_path):
    from valorant_voice.data.transcribe import WHISPER_RATE, whisper_input

    sf.write(tmp_path / "a.wav", np.zeros((48000, 2), dtype="float32"), 48000)

    samples = whisper_input(tmp_path / "a.wav")

    assert samples.ndim == 1 and abs(len(samples) - WHISPER_RATE) <= 1
