import numpy as np
import torch


class Normalizer:

    def __init__(self, cfg):

        self.cfg = cfg

        self.input_mean = None
        self.input_std = None

        self.output_mean = None
        self.output_std = None

    #########################################################################
    # FIT
    #########################################################################

    def fit(self, train_dataset):

        X_list = []
        Y_list = []

        print("Computing normalization statistics...")

        for x, y in train_dataset:

            x = x.numpy()
            y = y.numpy()

            ############################
            # INPUT PREPROCESSING
            ############################

            if self.cfg.preprocessing.input.log:

                # evita log(0)
                x = np.log(x + 1e-12)

            X_list.append(x)
            Y_list.append(y)

        X = np.stack(X_list)
        Y = np.stack(Y_list)

        ############################
        # INPUT STATS
        ############################

        self.input_mean = X.mean(axis=0)
        self.input_std = X.std(axis=0)

        self.input_std[self.input_std == 0] = 1.0

        ############################
        # OUTPUT STATS
        ############################

        self.output_mean = Y.mean(axis=0)
        self.output_std = Y.std(axis=0)

        self.output_std[self.output_std == 0] = 1.0

        print("Normalization statistics computed.")

    #########################################################################
    # INPUT
    #########################################################################

    def transform_input(self, x):

        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        if self.cfg.preprocessing.input.log:
            x = np.log(x + 1e-12)

        if self.cfg.preprocessing.input.normalization == "zscore":

            x = (x - self.input_mean) / self.input_std

        return torch.tensor(x, dtype=torch.float32)

    #########################################################################
    # OUTPUT
    #########################################################################

    def transform_output(self, y):

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        if self.cfg.preprocessing.output.normalization == "zscore":

            y = (y - self.output_mean) / self.output_std

        return torch.tensor(y, dtype=torch.float32)

    #########################################################################
    # INVERSE OUTPUT
    #########################################################################

    def inverse_output(self, y):

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        if self.cfg.preprocessing.output.normalization == "zscore":

            y = y * self.output_std + self.output_mean

        return torch.tensor(y, dtype=torch.float32)

    #########################################################################
    # SAVE
    #########################################################################

    def state_dict(self):

        return {

            "input_mean": self.input_mean,
            "input_std": self.input_std,

            "output_mean": self.output_mean,
            "output_std": self.output_std

        }

    #########################################################################
    # LOAD
    #########################################################################

    def load_state_dict(self, state):

        self.input_mean = state["input_mean"]
        self.input_std = state["input_std"]

        self.output_mean = state["output_mean"]
        self.output_std = state["output_std"]