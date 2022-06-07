import multiprocessing as mp

import hydra
from omegaconf import DictConfig, OmegaConf

from src.tools.run_pytorchtrainer import run_pytorchtrainer


@hydra.main(config_path="src/config/experiments", config_name="config")
def main(cfg: DictConfig) -> None:
    mp.set_start_method('spawn', force=True)
    print(OmegaConf.to_yaml(cfg))
    run_pytorchtrainer(cfg)


if __name__ == "__main__":
    main()

