from __future__ import annotations

import itertools
import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError("CSV must contain a 'target' column.")
    return df


def visualise_dataset(df: pd.DataFrame, max_cols: int = 4) -> None:
    y = df["target"].values
    n_x = len(df.columns) - 1
    n_rows = int(np.ceil(n_x / max_cols))

    fig, axes = plt.subplots(n_rows, max_cols, figsize=(4 * max_cols, 3.5 * n_rows))
    axes = axes.ravel()

    for i, col in enumerate(df.drop(columns="target").columns):
        axes[i].scatter(df[col], y, s=8, alpha=0.7)
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("target")

    for ax in axes[n_x:]:
        ax.axis("off")

    fig.suptitle("Q1 – explanatory variables vs target", fontsize=16, y=1.02)
    fig.tight_layout()
    plt.savefig('ds_features.pdf')
    plt.show()


def standardise_x(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = df.drop(columns="target").values
    y = df["target"].values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    return X_std, y, df.drop(columns="target").columns.to_list()


def generate_indicators(n_features: int) -> np.ndarray:
    combos = np.array(
        [np.array(list(np.binary_repr(i, width=n_features)), dtype=int)[::-1]
         for i in range(1, 2 ** n_features)],
        dtype=int,
    )
    return combos


def cross_val_cve(
    X: np.ndarray,
    y: np.ndarray,
    indicators: np.ndarray,
    model,
    n_splits: int = 10,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cve = np.empty(len(indicators))
    intercepts = np.empty(len(indicators))
    coefs = np.empty((len(indicators), X.shape[1]))

    for idx, ind in enumerate(indicators):
        mask = ind.astype(bool)
        mse_scores = []

        for tr, te in kf.split(X, y):
            model.fit(X[tr][:, mask], y[tr])
            pred = model.predict(X[te][:, mask])
            mse_scores.append(mean_squared_error(y[te], pred))

        model.fit(X[:, mask], y)
        intercepts[idx] = model.intercept_
        coefs[idx, mask] = model.coef_
        cve[idx] = np.mean(mse_scores)

    return cve, intercepts, coefs


def weight_diagram(coefs: np.ndarray, indicators: np.ndarray, title: str, save_path: str) -> None:
    mask = ~indicators.astype(bool)
    vmax = np.abs(coefs).max()
    cmap = LinearSegmentedColormap.from_list(
        "weight_cmap", [(0, "blue"), (0.5, "white"), (1, "red")]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(
        np.ma.masked_where(mask, coefs),
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        edgecolors="none",
    )
    ax.set_xlabel("Model rank (sorted by CVE)")
    ax.set_ylabel("Feature index")
    ax.set_title(title)
    plt.colorbar(mesh, ax=ax, label="Coefficient value")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data", type=Path, default=Path("boston.csv"), help="CSV path")
    p.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip scatter and weight diagrams (useful for CI).",
    )
    args = p.parse_args()

    df = load_dataset(args.data)
    print(f"Loaded dataset: {args.data}  shape={df.shape}")

    if not args.skip_plots:
        visualise_dataset(df)

    X_std, y, feature_names = standardise_x(df)
    n_features = X_std.shape[1]

    indicators = generate_indicators(n_features)
    print("\nQ4 – feature‑set indicators (1 means feature included):")
    print(indicators)
    print(len(indicators))

    models = {
        "Linear (with intercept)": LinearRegression(fit_intercept=True),
        "Linear (no intercept)": LinearRegression(fit_intercept=False),
        "Ridge α=1.0": Ridge(alpha=1.0, fit_intercept=True),
    }

    for tag, mdl in models.items():
        print(f"\n=== {tag} ===")
        cve, intercepts, coefs = cross_val_cve(X_std, y, indicators, mdl)

        order = np.argsort(cve)
        cve_sorted = cve[order]
        ind_sorted = indicators[order]
        coef_sorted = coefs[order]

        print(f"Best CVE: {cve_sorted[0]:.4f}")

        print("\nTop 5 models:")
        for i in range(5):
            print(f"  CVE={cve_sorted[i]:.4f}, indicator={ind_sorted[i]}")

        print("\nWorst 5 models:")
        for i in range(1, 6):
            print(f"  CVE={cve_sorted[-i]:.4f}, indicator={ind_sorted[-i]}")

        if args.skip_plots:
            continue

        ranks_to_plot = {
            "Top‑1 model": 1,
            "Top‑100 models": min(100, len(order)),
            "Top‑1000 models": min(1000, len(order)),
        }

        for ttl, k in ranks_to_plot.items():
            filename = f"{tag.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('.', '')}_{ttl.replace(' ', '_')}.pdf"
            weight_diagram(
                coef_sorted[:k].T,
                ind_sorted[:k].T,
                title=f"{tag} – {ttl}",
                save_path=filename,
            )
            print(f"Saved plot: {filename}")


if __name__ == "__main__":
    main()
