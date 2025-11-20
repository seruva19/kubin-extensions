from pathlib import Path

from omegaconf import DictConfig

from .kd5_config import KD5ConfigLoader
from .kd5_datasets import KD5RawDataset
from .kd5_trainer import Kandinsky5Trainer


def cache_kd5_assets(
    kubin,
    raw_config: DictConfig,
    *,
    build_latents: bool = True,
    build_text: bool = True,
    target_datasets=None,
    progress=None,
):
    loader = KD5ConfigLoader(Path(kubin.root))
    config = loader.load(raw_config)
    trainer = Kandinsky5Trainer(kubin, config, progress=progress)
    trainer.setup_components()

    # Filter datasets if target_datasets is specified
    datasets_to_cache = config.datasets
    if target_datasets is not None:
        if isinstance(target_datasets, str):
            target_datasets = [target_datasets]
        dataset_names = set(target_datasets)
        datasets_to_cache = [
            ds for ds in config.datasets
            if ds.name in dataset_names
        ]

    for dataset_cfg in datasets_to_cache:
        raw_dataset = KD5RawDataset(dataset_cfg)
        trainer.maybe_build_cache(
            dataset_cfg,
            raw_dataset,
            build_latents=build_latents,
            build_text=build_text,
        )


def launch_kd5_training(kubin, raw_config: DictConfig, progress=None):
    loader = KD5ConfigLoader(Path(kubin.root))
    config = loader.load(raw_config)
    trainer = Kandinsky5Trainer(kubin, config, progress=progress)
    trainer.train()


def precache_kd5_dataset(kubin, raw_config: DictConfig, progress=None):
    cache_kd5_assets(kubin, raw_config, build_latents=True, build_text=True, progress=progress)
