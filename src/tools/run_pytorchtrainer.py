import multiprocessing as mp
import os
from functools import partial

import hydra
import torch
from torch.utils.tensorboard import SummaryWriter

from src.data.build_dataloader import build_dataloader
from src.modeling.build_model import build_model
from src.tools.utils import *


def train_one_fold(cfg, val_id):
    train_loader, val_loader = build_dataloader(cfg, val_id)
    model = build_model(cfg)

    loss = hydra.utils.instantiate(cfg.loss_config,
                                   trans_rate=cfg.train_config.trans_rate,
                                   device=torch.device(cfg.train_config.device))

    trainer = hydra.utils.instantiate(cfg.train_config,
                                      criterion=loss,
                                      temp_dir=f"{os.getcwd()}/model/val_{val_id}",
                                      device=torch.device(cfg.train_config.device))

    tb = SummaryWriter()
    print(model)
    performance, inference = trainer.fit(model,
                                         train_loader=train_loader,
                                         val_loader=val_loader,
                                         test_loader=None,
                                         tensorboard_logger=tb)

    return performance, inference


def train_folds_parallel(cfg, n_val_fold):
    pool = mp.Pool(processes=2)
    func = partial(train_one_fold, cfg)
    result = pool.map(func, [i for i in range(n_val_fold)])
    pool.close()
    pool.join()

    performance_list = [result[i][0] for i in range(len(result))]
    inference_list = [result[i][1] for i in range(len(result))]

    return performance_list, inference_list


def train_folds_sequential(cfg, n_val_fold):
    performance_list = []
    inference_list = []
    for i in range(n_val_fold):
        performance, inference = train_one_fold(cfg, val_id=i)
        performance_list.append(performance)
        inference_list.append(inference)

    return performance_list, inference_list


def run_pytorchtrainer(cfg):
    n_val_fold = cfg.data_config.nb_validation_fold
    if cfg.run_in_parallel:
        for i_run in range(cfg.nb_runs):
            performance_list, inference_list = train_folds_parallel(cfg, n_val_fold)
            write_result(f'{os.getcwd()}/output', f'{i_run}', performance_list, inference_list)
    else:
        for i_run in range(cfg.nb_runs):
            performance_list, inference_list = train_folds_sequential(cfg, n_val_fold)
            write_result(f'{os.getcwd()}/output', f'{i_run}', performance_list, inference_list)

    average_run_results(f'{os.getcwd()}/output', nb_runs=cfg.nb_runs)
    median_run_results(f'{os.getcwd()}/output', nb_runs=cfg.nb_runs)


