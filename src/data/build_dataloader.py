import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def build_dataloader(cfg: DictConfig, val_id: int):
    train_dataset, val_dataset = build_dataset(cfg, val_id)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.dataloader_config.batch_size,
                              shuffle=cfg.dataloader_config.shuffle,
                              drop_last=cfg.dataloader_config.drop_last)

    val_loader = DataLoader(val_dataset,
                            batch_size=val_dataset.__len__(),
                            shuffle=cfg.dataloader_config.shuffle,
                            drop_last=cfg.dataloader_config.drop_last)

    return train_loader, val_loader


def build_dataset(cfg: DictConfig, val_id: int):
    train_dataset = hydra.utils.instantiate(cfg.feature_config,
                                            data_dir=cfg.data_config.data_path,
                                            index_range=cfg.data_config.folds_setup[val_id][0],
                                            dimension_order=cfg.model_config.dimension_order)

    val_dataset = hydra.utils.instantiate(cfg.feature_config,
                                          data_dir=cfg.data_config.data_path,
                                          index_range=cfg.data_config.folds_setup[val_id][1],
                                          dimension_order=cfg.model_config.dimension_order)

    return train_dataset, val_dataset

