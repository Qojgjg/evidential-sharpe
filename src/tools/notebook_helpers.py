import os
import pickle
from src.modeling.build_model import build_model
import torch
from src.modeling.solver.metrics import classification_metrics
from matplotlib import pyplot as plt


def load_last_model(cfg, temp_dir, model_dir):
    device = torch.device(cfg.train_config.device)

    checkpoint_files = [os.path.join(temp_dir + "/" + model_dir, f) for f in os.listdir(temp_dir + "/" + model_dir)
                            if f.startswith('checkpoint_')]

    fid = open(checkpoint_files[-1], 'rb')
    checkpoint = pickle.load(fid)
    fid.close()

    model = build_model(cfg)
    model.float()
    model.to(device)

    state_dict = checkpoint['model_state_dict']
    epoch_idx = checkpoint["epoch_idx"]

    model.load_state_dict(state_dict)

    return model, epoch_idx


def print_performance(performance, val_id, is_train):
    print(f"Fold: {val_id}", end="")
    for metric, result in performance.items():
        if is_train:
            print(f"| Train {metric}: {result:.4f}", end="")
        else:
            print(f"| Val {metric}: {result:.4f}", end="")
    print()


def forward(model, loader, logit_converter, device):
    evidence_list = []
    future_return_list = []
    label_list = []
    logit_list = []

    with torch.no_grad():
        for minibatch_idx, (data_batch, future_return_batch, label_batch) in enumerate(loader):
            data_batch = data_batch.to(device)
            future_return_batch = future_return_batch.to(device)
            label_batch = label_batch.to(device)

            logit = model(data_batch)
            evidence = logit_converter(logit)

            logit_list.append(logit)
            evidence_list.append(evidence)
            future_return_list.append(future_return_batch)
            label_list.append(label_batch)

        logit_full_batch = torch.cat(logit_list, 0)
        evidence_full_batch = torch.cat(evidence_list, 0)
        future_return_full_batch = torch.cat(future_return_list, 0)
        label_full_batch = torch.cat(label_list, 0)

    return evidence_full_batch, label_full_batch, future_return_full_batch, logit_full_batch


def compute_performance(evidence,
                        target,
                        future_return,
                        logit,
                        val_id,
                        metrics_func,
                        epoch_idx,
                        annealing_step,
                        is_train,
                        device):
    performance = {}
    for func_name, func in metrics_func.items():
        metric = func(evidence=evidence,
                      target=target,
                      future_return=future_return,
                      logit=logit,
                      epoch_idx=epoch_idx,
                      annealing_step=annealing_step)

        performance[func_name] = metric.item()

    print_performance(performance, val_id, is_train)


def histogram_uncertainty(evidence_train, label_train, evidence_test, label_test):
    u_train = classification_metrics.uncertainty_per_samples(evidence_train)
    match_vec_train = classification_metrics.match_vector(evidence_train, label_train)
    mask_train = match_vec_train == 1.0

    u_succ_train = torch.masked_select(u_train, mask_train)
    u_fail_train = torch.masked_select(u_train, torch.logical_not(mask_train))

    u_test = classification_metrics.uncertainty_per_samples(evidence_test)
    match_vec_test = classification_metrics.match_vector(evidence_test, label_test)
    mask_test = match_vec_test == 1.0

    u_succ_test = torch.masked_select(u_test, mask_test)
    u_fail_test = torch.masked_select(u_test, torch.logical_not(mask_test))


    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(u_succ_train.cpu().numpy(), bins=20, range=[1e-20, 1.0], color='blue',
                    alpha=0.7, rwidth=0.85)

    plt.hist(u_fail_train.cpu().numpy(), bins=20, range=[1e-20, 1.0], color='red',
                    alpha=0.7, rwidth=0.85)

    plt.subplot(1, 2, 2)
    plt.hist(u_succ_test.cpu().numpy(), bins=20, range=[1e-20, 1.0], color='blue',
                    alpha=0.7, rwidth=0.85)

    plt.hist(u_fail_test.cpu().numpy(), bins=20, range=[1e-20, 1.0], color='red',
                    alpha=0.7, rwidth=0.85)


def print_class_percentage(train_loader, val_loader, val_id):
    print(f"Train loader {val_id}: {train_loader.dataset.__get_class_percentage__()} | "
          f"Validation loader {val_id}: {val_loader.dataset.__get_class_percentage__()}")


def other_predictor_sharpe(pred_class, future_return, trans_rate, val_id, is_train):
    port_ret = (1 - trans_rate) * (1 + torch.gather(future_return, dim=1, index=pred_class)) * (
                1 - trans_rate) - 1

    assert future_return.shape[0] == torch.gather(future_return, dim=1, index=pred_class).shape[0]

    e_r = torch.mean(port_ret)
    std = torch.std(port_ret)
    sharpe = e_r / std

    name = "Val  "
    if is_train:
        name = "Train"
    print(f"{name} loader {val_id} | Sharpe ratio: {sharpe.item():.4f} | Expected return: {e_r.item():.4f} | "
          f"Standard deviation: {std.item():.4f}")