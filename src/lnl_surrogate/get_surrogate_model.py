import os
import random
import shutil
from glob import glob
from typing import List, Optional, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np

from .logger import logger
from .data_cache import DataCache
from .models import ModelType




def get_surrogate_model(
    model_dir: str,
    training_data_cache: Optional[DataCache] = None,
    gp_model=None,
    clean=False,
):
    """
    Get the ML surrogate model
    """
    if clean and gp_model.saved_model_exists(model_dir):
        shutil.rmtree(model_dir)

    if gp_model.saved_model_exists(model_dir):
        logger.info(f"Loading model from {model_dir}")
        model = gp_model.load(model_dir)
    else:
        param_data = training_data_cache.get_varying_params()
        tru_lnl = training_data_cache.true_lnl

        # ensure that the true lnl is a valid value
        if not np.isfinite(tru_lnl):
            raise ValueError("True lnl is not finite! Skipping analysis.")

        in_data = param_data.T
        out_data = training_data_cache.lnl.reshape(-1, 1)
        logger.info(
            f"Training model {model_dir}: IN[{in_data.shape}]--> OUT[{out_data.shape}]"
        )
        model = gp_model()
        plt_kwgs = dict(
            labels=training_data_cache.get_varying_param_keys(),
            truths=training_data_cache.true_param_vals.ravel(),
        )

        metrics = model.train(
            in_data, out_data, verbose=True, savedir=model_dir, extra_kwgs=plt_kwgs
        )
        logger.info(f"Surrogate metrics: {metrics}")

        # check if true lnl inside the range of pred_lnl
        pred_lnl = np.array(model(training_data_cache.true_param_vals)).ravel()
        check = (pred_lnl[0] <= tru_lnl) & (tru_lnl <= pred_lnl[2])
        diff = np.max(
            [np.abs(pred_lnl[1] - pred_lnl[0]), np.abs(pred_lnl[1] - pred_lnl[2])]
        )
        pred_str = f"{pred_lnl[1]:.2f} +/- {diff:.2f}"
        check_str = "✔" * 3 if check else "❌" * 3
        logger.info(
            f"{check_str} True LnL: {tru_lnl:.2f}, Surrogate LnL: {pred_str} {check_str}"
        )
    logger.success("Trained and saved Model")

    return model
