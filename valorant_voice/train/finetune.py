"""Fine-tunes the XTTS-v2 GPT on the pooled train manifest of every agent.

Needs a GPU with about 12 GB or more: run it on a free Kaggle or Colab session, not the 6 GB laptop
GPU. Copy `data/clean/` and `data/manifests/` to the session, run this file, then copy the output
folder back and point VV_CHECKPOINT_DIR at it. The validation manifest is used for monitoring; the
test manifest is never read here.

    python -m valorant_voice.train.finetune --epochs 6 --batch-size 3 --grad-accum 5

The result is exported to <output>/model, a folder VV_CHECKPOINT_DIR can point at.
"""

import argparse
import shutil
from pathlib import Path

from .. import config

PROJECT = "valorant-voice-ai"


BASE_FILES = ("vocab.json", "mel_stats.pth", "dvae.pth")


def export_checkpoint(run_dir: Path, base: Path, target: Path) -> Path:
    """Builds the folder XttsEngine loads: the newest trained weights plus the base model's support files."""
    weights = sorted(run_dir.rglob("*.pth"), key=lambda p: p.stat().st_mtime)
    weights = [p for p in weights if p.name in ("best_model.pth",) or p.name.startswith("checkpoint_")]
    if not weights:
        raise FileNotFoundError(f"No trained weights under {run_dir}")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(weights[-1], target / "model.pth")
    shutil.copy(base / "config.json", target / "config.json")
    for name in BASE_FILES:
        if (base / name).exists():
            shutil.copy(base / name, target / name)
    return target


def finetune(epochs: int, batch_size: int, grad_accum: int, learning_rate: float, output: Path) -> Path:
    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.xtts_config import XttsAudioConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
    from TTS.utils.manage import ModelManager

    from ..tts.engine import BASE_MODEL

    base = Path(ModelManager().download_model(BASE_MODEL)[1]).parent
    dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="valorant-agents",
        path=str(config.CLEAN_DIR),
        meta_file_train=str(config.MANIFEST_DIR / "train.csv"),
        meta_file_val=str(config.MANIFEST_DIR / "val.csv"),
        language="en",
    )
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 s of reference audio at 22.05 kHz
        min_conditioning_length=66150,  # 3 s
        max_wav_length=255995,  # about 11.6 s
        max_text_length=200,
        mel_norm_file=str(base / "mel_stats.pth"),
        dvae_checkpoint=str(base / "dvae.pth"),
        xtts_checkpoint=str(base / "model.pth"),
        tokenizer_file=str(base / "vocab.json"),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    train_config = GPTTrainerConfig(
        output_path=str(output),
        model_args=model_args,
        run_name="xtts-agents",
        project_name=PROJECT,
        dashboard_logger="tensorboard",
        audio=XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000),
        batch_size=batch_size,
        batch_group_size=48,
        eval_batch_size=batch_size,
        num_loader_workers=4,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=2000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=learning_rate,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},  # far beyond this run: constant rate
        epochs=epochs,
        test_sentences=[],
    )
    train_samples, eval_samples = load_tts_samples([dataset], eval_split=False)
    trainer = Trainer(
        TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=True, grad_accum_steps=grad_accum),
        train_config,
        output_path=str(output),
        model=GPTTrainer.init_from_config(train_config),
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
    return export_checkpoint(output, base, output / "model")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=5, help="effective batch = batch-size * grad-accum")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--output", type=Path, default=config.ROOT / "checkpoints")
    args = parser.parse_args()
    print("exported model folder:", finetune(args.epochs, args.batch_size, args.grad_accum, args.lr, args.output))


if __name__ == "__main__":
    main()
