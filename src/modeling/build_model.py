import hydra
from src.tools.convert_logits import relu_evidence, exp_evidence, softplus_evidence


def build_model(cfg):

    output_dim = [cfg.data_config.allocation_asset, 1]
    lb_wn_size = cfg.feature_config.lb_wn_size

    model = hydra.utils.instantiate(cfg.model_config,
                                    spatial_dim=cfg.feature_config.nb_total_feature,
                                    temporal_dim=lb_wn_size,
                                    output_dim=output_dim,
                                    normalization_scheme=cfg.feature_config.normalization_scheme,
                                    device=cfg.train_config.device)

    return model


