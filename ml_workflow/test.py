import torch

from test_utils import test

from hydra.utils import instantiate
from omegaconf import OmegaConf

from dataset.dataset import make_datasets, make_dataloaders
from preprocessing.normalizer import Normalizer


def run_test(cfg, checkpoint_path = None):

    ###########################################################
    # LOAD CHECKPOINT
    ###########################################################

    if checkpoint_path is None:
        checkpoint_path = cfg.test.checkpoint

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only = False)


    train_cfg = OmegaConf.create(checkpoint["cfg"])

    ###########################################################
    # MODEL
    ###########################################################

    model = instantiate(train_cfg.model)

    model.load_state_dict(checkpoint["model"])


    ##############

    criterion = instantiate(train_cfg.loss)

    ###########################################################
    # NORMALIZER
    ###########################################################

    normalizer = Normalizer(train_cfg)

    normalizer.load_state_dict(checkpoint["normalizer"])

    ###########################################################
    # DATASET
    ###########################################################

    dataset, train_dataset, val_dataset, test_dataset = make_datasets(train_cfg)

    dataset.normalizer = normalizer

    ###########################################################
    # DATALOADER
    ###########################################################

    _, _, test_loader = make_dataloaders(
        train_cfg,
        train_dataset,
        val_dataset,
        test_dataset,
    )

    ###########################################################
    # TEST
    ###########################################################

    test(
        cfg=train_cfg,
        model=model,
        criterion=criterion,
        test_loader=test_loader,
        normalizer=normalizer,
    )