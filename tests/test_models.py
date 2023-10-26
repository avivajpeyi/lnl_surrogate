import os
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import scipy

np.random.seed(1)

import pytest

from lnl_surrogate.models import load_model

FUNCTIONS = dict(
    curvy=lambda x: 0.2 + 0.4 * x ** 2 + 0.3 * x * np.sin(15 * x) + 0.05 * np.cos(50 * x),
    wavelet=lambda x: scipy.stats.norm(0.5, 0.15).pdf(x) * np.sin(50 * x),
)


def train_and_save_model(model_class, data: np.ndarray, model_path: str):
    model = model_class()
    model.train(data[0], data[1], verbose=True, savedir=model_path)
    preds = model.predict(data[0])
    loaded_model = model_class.load(model_path)
    loaded_preds = loaded_model.predict(data[0])
    assert np.allclose(preds, loaded_preds)


def plot(ax, true_f, models, model_names):
    data = generate_data(true_f, 500)
    x = data[0]
    y = true_f(x)
    ax.plot(x, y, label="True", color="black", ls="--", zorder=10)
    for i, model in enumerate(models):
        low_y, mean_y, up_y = model(x)
        ax.plot(
            x,
            mean_y,
            label=f"{model_names[i]} (R^2:{model.r_sqred(data):.2f})",
            color=f"C{i}",
            alpha=0.5,
        )
        ax.fill_between(
            x.flatten(),
            low_y.flatten(),
            up_y.flatten(),
            alpha=0.1,
            color=f"C{i}",
        )
    ax.legend(frameon=False)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.axis("off")


def generate_data(func: Callable, n=50) -> np.ndarray:
    x = np.random.uniform(0, 1, n)
    x = np.sort(x)
    y = func(x)
    X = [[i] for i in x]
    Y = [[i] for i in y]
    return np.array([X, Y])


@pytest.mark.parametrize(
    "model_type, n_training_pts, func_name",
    [
        ("gpflow", [10, 25, 50], "curvy"),
        # ("sklearngp", [10, 25, 50], "curvy")
    ],
)
def test_models(tmpdir, model_type: str, n_training_pts: List[int], func_name: str):
    np.random.seed(0)
    outdir = f"{tmpdir}/test_models"
    os.makedirs(outdir, exist_ok=True)

    model_class = load_model(model_type)

    # get some names for the model and function
    model_name = model_class.__name__
    func_to_learn = FUNCTIONS[func_name]
    paths = [f"{outdir}/{model_name}_{func_name}_model_n{pts}" for pts in n_training_pts]

    # TRAIN and save the model
    for pts, pth in zip(n_training_pts, paths):
        train_and_save_model(
            model_class,
            generate_data(func_to_learn, pts),
            pth
        )

    # LOAD plot the model for a visual check
    fig = plt.figure(figsize=(5, 3))
    plot(
        fig.gca(),
        true_f=func_to_learn,
        models=[model_class.load(pth) for pth in paths],
        model_names=[f"{pts} Training pts" for pts in n_training_pts],
    )
    plt.tight_layout()
    plt.savefig(f"{outdir}/{model_name}.png")
