import os
import torch

from train_utils import train

from hydra.utils import instantiate

from dataset.dataset import make_datasets, make_dataloaders
from preprocessing.normalizer import Normalizer


def run_train(cfg):

    ###########################################################
    # DATASET
    ###########################################################

    dataset, train_dataset, val_dataset, test_dataset = make_datasets(cfg)

    ###########################################################
    # NORMALIZER
    ###########################################################

    normalizer = Normalizer(cfg)
    normalizer.fit(dataset)

    dataset.normalizer = normalizer

    ###########################################################
    # DATALOADERS
    ###########################################################

    train_loader, val_loader, _ = make_dataloaders(
        cfg,
        train_dataset,
        val_dataset,
        test_dataset,
    )

    ###########################################################
    # MODEL
    ###########################################################

    model = instantiate(cfg.model)

    ###########################################################
    # TRAIN
    ###########################################################

    best_checkpoint = train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
    )

    return best_checkpoint