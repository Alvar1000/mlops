import hydra
from omegaconf import DictConfig
from matching.train import train as train_func


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    train_func(cfg)


if __name__ == "__main__":
    main()
