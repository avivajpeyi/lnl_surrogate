from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..data_cache import DataCache
from ..logger import logger
from ..plotting import plot_model_diagnostics
from .model_metrics import ModelMetrics
from .utils import fmt_val_upper_lower


class Model(ABC, ModelMetrics):
    """Base class for surrogate models."""

    def __init__(self):
        self._model = None
        self.trained = False
        self.input_dim = None
        # self.scaler = StandardScaler()
        self.scaler = None

    @staticmethod
    @abstractmethod
    def saved_model_exists(savedir: str) -> bool:
        """Check if a model exists in the given directory."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, savedir: str) -> "Model":
        """Load a model from a dir."""
        pass

    @abstractmethod
    def save(self, savedir: str) -> None:
        """Save a model to a dir."""
        pass

    @abstractmethod
    def train(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        unc: Optional[np.ndarray] = None,  # output uncertainties
        verbose: Optional[bool] = False,
        savedir: Optional[str] = None,
    ) -> Dict[str, "Metrics"]:
        """Train the model.

        :return Dict[str, Metrics]: metrics for training and testing
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output of the model for the given input. (lower, mean, upper)"""
        pass

    def _post_training(
        self,
        training_data: DataCache,
        testing_data: DataCache,
        savedir,
        extra_kwgs={},
    ):
        """Post training processing."""
        self.trained = True
        self.input_dim = training_data[0].shape[1]
        if savedir:
            self.save(savedir)
            self.plot_diagnostics(
                training_data, testing_data, savedir, extra_kwgs
            )
        preds = self.predict(training_data[0])[1]
        # check that the predictions are not all zero
        if np.allclose(preds, 0):
            raise ValueError("All predictions are zero")
        return self.train_test_metrics(training_data, testing_data)

    def fit(self, inputs, outputs):
        """Alias for train."""
        return self.train(inputs, outputs)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def prediction_str(self, x: np.ndarray) -> Union[str, List[str]]:
        """Format prediction as a latex string with error bars."""
        lower, mean, upper = self.predict(x)
        q0, q1 = np.abs(np.round(lower - mean, 2)), np.abs(
            np.round(mean - upper, 2)
        )
        t, b = np.maximum(q0, q1), np.minimum(q0, q1)
        m = np.round(mean, 2)
        strs = [fmt_val_upper_lower(m, t, b) for m, t, b in zip(m, t, b)]
        if len(strs) == 1:
            return strs[0]
        else:
            return strs

    def _preprocess_input(self, inputs):
        """Preprocess the input."""
        # return self.scaler.transform(inputs)
        return inputs

    def _preprocess_and_split_data(
        self, input, output, unc=None, test_size=0.2
    ):
        """
        Preprocess and split data into training and testing sets.
        :param input:
        :param output:
        :param test_size:
        :return: (train_in, test_in, train_out, test_out)
        """

        if unc is not None:
            output = np.hstack((output, unc))

        # self.scaler = StandardScaler()
        # input_scaled = self.scaler.fit_transform(input)
        input_scaled = input

        # check shape of input and output are the same
        if input_scaled.shape[0] != output.shape[0]:
            raise ValueError(
                f"Input ({input_scaled.shape}) and output ({output.shape}) "
                "must have the same number of samples"
            )

        # check that input and output are tensors (len(shape) > 1)
        if len(input_scaled.shape) < 2 or len(output.shape) < 2:
            raise ValueError(
                "Input and output must be tensors (len(shape) > 1)"
            )

        (train_in, test_in, train_out, test_out) = train_test_split(
            input_scaled, output, test_size=test_size
        )

        if unc is not None:
            train_unc, train_out = train_out[:, -1], train_out[:, :-1]
            test_unc, test_out = test_out[:, -1], test_out[:, :-1]
            train_unc = train_unc.reshape(-1, 1)
            test_unc = test_unc.reshape(-1, 1)
        else:
            train_unc, test_unc = None, None

        logger.info(
            f"Training surrogate In({train_in.shape})-->Out({train_out.shape}) [testing:{len(test_out)}]"
        )

        return train_in, test_in, train_out, test_out, train_unc, test_unc

    def plot_diagnostics(self, train_data, test_data, savedir, extra_kwgs={}):
        """Plot diagnostics for the model."""
        plot_model_diagnostics(
            self, train_data, test_data, savedir, extra_kwgs
        )
