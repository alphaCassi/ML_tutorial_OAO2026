import hydra

from train import run_train
from test import run_test


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):

    checkpoint = None

    if cfg.mode.train:
        checkpoint = run_train(cfg)

    if cfg.mode.test:
        run_test(cfg, checkpoint)

if __name__ == "__main__":
    main()