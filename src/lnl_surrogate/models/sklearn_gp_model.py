import os
import pickle
from typing import Optional, Tuple

import numpy as np
from scipy.stats import halfnorm
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import r2_score

from .base_model import Model

MODEL_SAVE_FILE = "model.pkl"


class SklearnGPModel(Model):
    """Scikit-learn GP (Gaussian process) surrogate model"""

    def __init__(self):
        super().__init__()
        # # diff between every pair of train_out values
        # err = np.min(np.diff(train_out, axis=0) ** 2)

        # alpha:
        # It can also be interpreted as the variance of additional
        # Gaussian measurement noise on the training observations.

    def train(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        unc: Optional[np.ndarray] = None,  # output uncertainties
        verbose: Optional[bool] = False,
        savedir: Optional[str] = None,
        extra_kwgs={},
    ) -> None:
        """Train the model.

        https://scikit-learn.org/dev/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
        GaussianProcessRegressor(
            kernel=None, *, alpha=1e-10,
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=0,
            normalize_y=False, copy_X_train=True,
            n_targets=None, random_state=None
        )

        """
        (
            train_in,
            test_in,
            train_out,
            test_out,
            train_unc,
            test_unc,
        ) = self._preprocess_and_split_data(inputs, outputs, unc)

        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        if unc is not None:
            alpha = ((train_unc / train_out) ** 2).ravel()
        else:
            alpha = 1e-10
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, n_restarts_optimizer=100
        )
        gp.fit(train_in, train_out)
        self._model = gp

        return self._post_training(
            (train_in, train_out), (test_in, test_out), savedir, extra_kwgs
        )

    def predict(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict the output of the model for the given input."""
        # x_scaled = self._preprocess_input(x)
        x_scaled = x  # TODO: bug with scaling -- i might be scaling twice
        y, y_std = self._model.predict(x_scaled, return_std=True)
        y_lower = y - 1.96 * y_std
        y_upper = y + 1.96 * y_std
        return y_lower, y, y_upper

    def save(self, savedir: str) -> None:
        """Save a model to a dir."""
        if not self.trained:
            raise ValueError("Model not trained, no point saving")
        os.makedirs(savedir, exist_ok=True)
        with open(f"{savedir}/{MODEL_SAVE_FILE}", "wb") as f:
            pickle.dump([self._model, self.scaler], f)

    @classmethod
    def load(cls, savedir: str) -> "Model":
        """Load a model from a dir."""
        with open(cls.model_fn(savedir), "rb") as f:
            loaded_model = pickle.load(f)
        model = cls()
        model._model = loaded_model[0]
        model.scaler = loaded_model[1]
        model.trained = True
        return model

    @staticmethod
    def saved_model_exists(savedir: str) -> bool:
        """Check if a model exists in the given directory."""
        return os.path.exists(SklearnGPModel.model_fn(savedir))

    @staticmethod
    def model_fn(savedir: str):
        return f"{savedir}/{MODEL_SAVE_FILE}"

    def get_model(self):
        return self._model
