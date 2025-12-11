# fastai 2.x nonlinear regression demo
# pip install fastai matplotlib

from fastai.tabular.all import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. Target function and synthetic data
# --------------------------------------------------------

def true_function(x0, x1):
    "Ground truth nonlinear function f(x0, x1)."
    return np.exp(-((x0 - 0.5)**2 + (x1 + 0.5)**2)) \
           + 0.5 * np.sin(np.pi * x0) * np.sin(np.pi * x1)

def make_synthetic_df(n=10_000, noise=0.05, seed=42):
    "Generate synthetic regression data in [-2, 2]^2 with optional Gaussian noise."
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-2.0, 2.0, size=n)
    y  = true_function(x0, x1)
    if noise and noise > 0:
        y += rng.normal(0.0, noise, size=n)
    df = pd.DataFrame({'x0': x0, 'x1': x1, 'y': y})
    return df

# --------------------------------------------------------
# 2. DataLoaders
# --------------------------------------------------------

def make_dls(df, valid_pct=0.2, bs=256):
    """
    Build fastai TabularDataLoaders for regression.
    All features are continuous, target is y (float).
    """
    cont_names = ['x0', 'x1']
    y_names    = 'y'
    splits = RandomSplitter(valid_pct=valid_pct, seed=0)(df)

    to = TabularPandas(
        df,
        procs=[Normalize],        # normalize continuous vars
        cat_names=[],             # no categorical features
        cont_names=cont_names,
        y_names=y_names,
        y_block=RegressionBlock(),  # float target
        splits=splits
    )

    dls = to.dataloaders(bs=bs)
    return dls, to

# --------------------------------------------------------
# 3. Learner
# --------------------------------------------------------

def make_learner(dls, layers=(64, 64), y_range=None):
    """
    Create a tabular_learner for regression.
    layers: hidden layer sizes for the MLP
    y_range: optional [min, max] clamp for outputs (can be None).
    """
    learn = tabular_learner(
        dls,
        layers=list(layers),
        y_range=y_range,
        loss_func=MSELossFlat(),  # regression loss
        metrics=rmse              # root mean squared error
    )
    return learn

# --------------------------------------------------------
# 4. Training wrapper
# --------------------------------------------------------

def train_model(learn, epochs=50, lr=1e-2):
    "Train the model with one_cycle policy."
    learn.fit_one_cycle(epochs, lr)
    return learn

# --------------------------------------------------------
# 5. Plots
# --------------------------------------------------------

def plot_training_loss(learn):
    "Plot training and validation loss over epochs."
    learn.recorder.plot_loss()
    plt.title("Training and validation loss")
    plt.savefig('fastai_training_loss.png')
    plt.show()

def _grid_df(xmin=-2, xmax=2, ymin=-2, ymax=2, steps=80):
    "Helper: make a grid dataframe over the 2D input space."
    xs = np.linspace(xmin, xmax, steps)
    ys = np.linspace(ymin, ymax, steps)
    X0, X1 = np.meshgrid(xs, ys)
    flat = pd.DataFrame({'x0': X0.ravel(), 'x1': X1.ravel()})
    return X0, X1, flat

def plot_true_vs_pred_surface(learn, steps=80):
    """
    Show heatmaps of:
      left: true function
      right: network prediction
    over [-2, 2] x [-2, 2].
    """
    X0, X1, flat = _grid_df(steps=steps)

    # true surface
    Z_true = true_function(X0, X1)

    # predicted surface
    dl = learn.dls.test_dl(flat)           # no labels, only x0,x1
    preds, _ = learn.get_preds(dl=dl)
    Z_pred = preds.detach().cpu().numpy().reshape(X0.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(
        Z_true,
        origin='lower',
        extent=[-2, 2, -2, 2]
    )
    axes[0].set_title("True f(x0, x1)")
    axes[0].set_xlabel("x0")
    axes[0].set_ylabel("x1")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        Z_pred,
        origin='lower',
        extent=[-2, 2, -2, 2]
    )
    axes[1].set_title("Predicted f̂(x0, x1)")
    axes[1].set_xlabel("x0")
    axes[1].set_ylabel("x1")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig('fastai_true_vs_pred_surface.png')
    plt.show()

# --------------------------------------------------------
# 6. Example main (for your notebook or script)
# --------------------------------------------------------

if __name__ == "__main__":
    # 1) data
    df = make_synthetic_df(n=10_000, noise=0.05)

    # 2) DataLoaders
    dls, to = make_dls(df, valid_pct=0.2, bs=256)

    # 3) learner
    # y_range is optional; fastai handles pure regression with None.
    learn = make_learner(dls, layers=(64, 64), y_range=None)

    # 4) train
    learn = train_model(learn, epochs=50, lr=1e-2)

    # 5) diagnostics
    plot_training_loss(learn)
    plot_true_vs_pred_surface(learn, steps=80)
