import torch
from tqdm import tqdm

from src.modeling.solver.metrics import classification_metrics, trading_metrics
from src.tools.PyTorchTrainer import ClassifierTrainer, get_cosine_lr_scheduler
from src.tools.convert_logits import relu_evidence


class DLSTrainer(ClassifierTrainer):
    def __init__(self,
                 n_epoch,
                 criterion,
                 trans_rate,
                 start_lr,
                 end_lr,
                 device,
                 epoch_idx=0,
                 lr_scheduler=get_cosine_lr_scheduler,
                 optimizer='adam',
                 weight_decay=1e-4,
                 temp_dir='',
                 checkpoint_freq=1,
                 print_freq=1,
                 use_progress_bar=True,
                 test_mode=False):

        super(DLSTrainer, self).__init__(n_epoch,
                                         epoch_idx,
                                         lr_scheduler,
                                         optimizer,
                                         weight_decay,
                                         temp_dir,
                                         checkpoint_freq,
                                         print_freq,
                                         use_progress_bar,
                                         test_mode)

        self.criterion = criterion
        self.lr_scheduler = lr_scheduler(start_lr, end_lr)
        self.logit_converter = relu_evidence
        self.trans_rate = trans_rate
        self.device = device
        self.metrics = ['sharpe_ratio', 'expected_return', 'standard_deviation', 'mean_cost', 'maximum_drawdown']
        self.metrics_func = {
            "Squared Error Bayes Risk": classification_metrics.squared_error_bayes_risk,
            "KL Divergence": classification_metrics.kl_divergence_loss,
            "EDL Loss": classification_metrics.edl_loss,
            "Mean Evidence Success": classification_metrics.mean_evidence_succ,
            "Mean Evidence Fail": classification_metrics.mean_evidence_fail,
            "Mean Uncertainty Success": classification_metrics.mean_uncertainty_succ,
            "Mean Uncertainty Fail": classification_metrics.mean_uncertainty_fail,
            "Accuracy": classification_metrics.accuracy,
            "Cross Entropy": classification_metrics.cross_entropy_loss,
            "Sharpe Ratio": trading_metrics.cal_sharpe_ratio
        }
        self.monitor_metric = "Squared Error Bayes Risk"
        self.monitor_direction = 'lower'

    def update_loop(self, model, loader, optimizer, device):
        if self.test_mode:
            total_minibatch = min(self.n_test_minibatch, len(loader))
        else:
            total_minibatch = len(loader)

        minibatch_idx = 0

        if self.use_progress_bar:
            loader = tqdm(loader, desc='#Epoch {}/{}: '.format(self.epoch_idx + 1, self.n_epoch), ncols=80, ascii=True)
        else:
            loader = loader

        for data_batch, future_return_batch, labels_batch in loader:
            optimizer.zero_grad()

            data_batch = data_batch.to(device)
            future_return_batch = future_return_batch.to(device)
            labels_batch = labels_batch.to(device)

            logit = model(data_batch)
            evidence = self.logit_converter(logit)

            loss = self.criterion(evidence=evidence,
                                  target=labels_batch,
                                  logit=logit,
                                  epoch_idx=self.epoch_idx)

            loss.backward()
            optimizer.step()

            minibatch_idx += 1

            if minibatch_idx > total_minibatch:
                break

    def eval(self, model, loader, device=torch.device("cuda")):
        if loader is None:
            return {}

        model.eval()

        logit_list = []
        evidence_list = []
        future_return_list = []
        label_list = []

        if self.test_mode:
            total_minibatch = min(self.n_test_minibatch, len(loader))
        else:
            total_minibatch = len(loader)

        with torch.no_grad():
            for minibatch_idx, (data_batch, future_return_batch, label_batch) in enumerate(loader):
                if minibatch_idx == total_minibatch:
                    break

                data_batch = data_batch.to(device)
                future_return_batch = future_return_batch.to(device)
                label_batch = label_batch.to(device)

                logit = model(data_batch)
                evidence = self.logit_converter(logit)

                logit_list.append(logit)
                evidence_list.append(evidence)
                future_return_list.append(future_return_batch)
                label_list.append(label_batch)

            logit_full_batch = torch.cat(logit_list, 0)
            evidence_full_batch = torch.cat(evidence_list, 0)
            future_return_full_batch = torch.cat(future_return_list, 0)
            label_full_batch = torch.cat(label_list, 0)

            performance = {}
            for func_name, func in self.metrics_func.items():
                metric = func(evidence=evidence_full_batch,
                              target=label_full_batch,
                              logit=logit_full_batch,
                              future_return=future_return_full_batch,
                              epoch_idx=self.epoch_idx,
                              annealing_step=self.criterion.annealing_step,
                              trans_rate=self.trans_rate,
                              device=self.device)

                performance[func_name] = metric.item()

        return performance

    def inference(self, model, loader, device):
        if loader is None:
            return {}

        model.eval()

        logits_list = []
        future_return_list = []

        if self.test_mode:
            total_minibatch = min(self.n_test_minibatch, len(loader))
        else:
            total_minibatch = len(loader)

        with torch.no_grad():
            for minibatch_idx, (data_batch, future_return_batch, label_batch) in enumerate(loader):
                if minibatch_idx == total_minibatch:
                    break

                data_batch = data_batch.to(self.device)
                future_return_batch = future_return_batch.to(self.device)

                logits = model(data_batch)
                logits_list.append(logits)

                future_return_list.append(future_return_batch)

            logits_full_batch = torch.cat(logits_list, 0)
            future_return_full_batch = torch.cat(future_return_list, 0)

        inference = {'logits': logits_full_batch.cpu(),
                     'future_return': future_return_full_batch.cpu()}

        return inference
