import numpy as np
import soundfile as sf

from valorant_voice.data.split import assign_split
from valorant_voice.data.text import text_key
from valorant_voice.tts.references import build_references, choose_references


def test_choose_references_prefers_clips_near_six_seconds():
    clips = [("a", 2.0), ("b", 5.5), ("c", 6.2), ("d", 11.9), ("e", 4.0), ("f", 30.0)]
    assert choose_references(clips, count=3) == ["c", "b", "e"]


def test_choose_references_returns_nothing_when_no_clip_fits():
    assert choose_references([("a", 2.0), ("b", 20.0)]) == []


def test_build_references_only_uses_train_split_lines(tmp_path):
    clean, out = tmp_path / "clean", tmp_path / "reference"
    (clean / "sage").mkdir(parents=True)
    rows = []
    for i in range(200):
        sf.write(clean / "sage" / f"c{i}.wav", np.zeros(22050 * 5, dtype="float32"), 22050)
        rows.append(f"c{i}|Line number {i}")
    (clean / "sage" / "transcripts.csv").write_text("\n".join(rows), encoding="utf-8")

    chosen = build_references("sage", clean_dir=clean, out_dir=out)

    assert len(chosen) == 3
    assert all(assign_split(text_key(f"Line number {c[1:]}")) == "train" for c in chosen)


def test_export_checkpoint_copies_newest_weights_and_base_files(tmp_path):
    import os

    from valorant_voice.train.finetune import export_checkpoint

    base, run = tmp_path / "base", tmp_path / "run" / "xtts-agents-1"
    base.mkdir()
    run.mkdir(parents=True)
    for name in ("config.json", "vocab.json", "mel_stats.pth", "dvae.pth"):
        (base / name).write_text(name)
    old, new = run / "checkpoint_100.pth", run / "best_model.pth"
    old.write_text("old")
    new.write_text("new")
    os.utime(old, (1, 1))

    folder = export_checkpoint(tmp_path / "run", base, tmp_path / "out")

    assert (folder / "model.pth").read_text() == "new"
    assert {p.name for p in folder.iterdir()} == {"model.pth", "config.json", "vocab.json", "mel_stats.pth", "dvae.pth"}


def test_support_file_prefers_the_base_folder(tmp_path):
    from valorant_voice.train.finetune import support_file

    (tmp_path / "dvae.pth").write_text("x")

    assert support_file(tmp_path, "dvae.pth") == str(tmp_path / "dvae.pth")
