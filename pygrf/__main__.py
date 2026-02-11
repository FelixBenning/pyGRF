import os
import polars as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from pygrf.grf import IsotropicGRF
from pygrf.kernels import SquaredExponentialKernel

# Use Computer Modern-like fonts: set serif family and mathtext to use cm
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        # don't require an external LaTeX installation by leaving usetex False
        "text.usetex": True,
    }
)


def simulate_gd(dim, steps, *, x0=None, rng=None, learning_rate: float = 1.0):
    """simulate GD on a GRF, return the function values along the trajectory

    Parameters
    - x0: initial point (optional)
    - dim: problem dimension
    - steps: number of GD steps
    - rng: numpy Generator (optional)
    - learning_rate: multiplier applied to the gradient when stepping
    """
    rng = np.random.default_rng() if rng is None else rng
    if x0 is None:
        x0 = rng.standard_normal(dim)
    grf = IsotropicGRF(dim=dim, kernel=SquaredExponentialKernel(), rng=rng)
    x0 = grf.into_adapted_span(x0)
    locs = [x0]
    f_vals = np.empty(steps)
    for t in range(steps):
        x = locs[-1]
        f_vals[t], grad = grf(x, with_gradient=True)
        locs.append(x - learning_rate * grad)

    return f_vals


class SimulationCache:
    """Context manager for a single-parquet simulation cache.

    Usage:
        with SimulationCache(path) as cache:
            # read via cache.df, modify and assign back to cache.df
            cache.df = ensure_simulations(cache.df, dim, repeats, steps)
    When the context exits the DataFrame is written back to `path`.
    """

    def __init__(self, path: str):
        self.path = path
        self.df = None

    def __enter__(self):
        if os.path.exists(self.path):
            self.df = pl.read_parquet(self.path)
        else:
            # empty DataFrame with expected columns (include learning rate 'lr')
            self.df = pl.DataFrame(
                {"dim": [], "seed": [], "steps": [], "lr": [], "f_vals": []}
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        # always write the current dataframe back to disk
        # convert to parquet; overwrite existing file
        # ensure df exists
        if self.df is None:
            return
        self.df.write_parquet(self.path)


def ensure_simulations(
    df: pl.DataFrame, dim: int, repeats: int, steps: int, learning_rate: float = 1.0
) -> pl.DataFrame:
    """Ensure the DataFrame contains `repeats` simulation runs for `dim` and
    `learning_rate`.

    Existing runs are matched by both `dim` and `lr` (learning rate). Any
    rows for the same `dim`/`lr` with fewer than `steps` entries in `f_vals`
    are removed and re-simulated.
    """
    # select existing rows for this dim and learning rate with sufficient steps
    existing_valid = df.filter(
        (pl.col("dim") == dim)
        & (pl.col("lr") == learning_rate)
        & (pl.col("steps") >= steps)
    )

    valid_seeds = (
        set(existing_valid["seed"].to_list()) if existing_valid.height > 0 else set()
    )

    # remove rows for this dim+lr that have insufficient steps
    df = df.filter(
        ~(
            (pl.col("dim") == dim)
            & (pl.col("lr") == learning_rate)
            & (pl.col("steps") < steps)
        )
    )

    # if already have enough valid runs, return
    if len(valid_seeds) >= repeats:
        return df

    needed = set(range(repeats))
    missing = sorted(list(needed - valid_seeds))

    records = []
    for seed in missing:
        rng = np.random.default_rng(seed)
        x0 = rng.standard_normal(dim)
        f_vals = simulate_gd(dim, steps, x0=x0, rng=rng, learning_rate=learning_rate)
        records.append(
            {
                "dim": dim,
                "seed": seed,
                "steps": steps,
                "lr": learning_rate,
                "f_vals": f_vals,
            }
        )

    if records:
        newdf = pl.DataFrame(records)
        if df.height == 0:
            df = newdf
        else:
            df = pl.concat([df, newdf], how="vertical")

    return df


# Use the cache context manager and ensure simulations exist for dims
def plot_gd_trajectory(
    ax,
    cache: "SimulationCache",
    dim: int,
    repeats: int,
    steps: int,
    learning_rate: float = 1.0,
):
    """Ensure simulations for `dim` and draw trajectories + mean/std on `ax`.

    Returns the final values array for this dimension (shape: (n_repeats,)).
    """
    cache.df = ensure_simulations(cache.df, dim, repeats, steps, learning_rate)
    subset = cache.df.filter(pl.col("dim") == dim).sort("seed")
    f_vals = np.stack(subset["f_vals"].to_list())

    ax.set_title(f"GD trajectory (dim={dim})")
    for repeat in range(f_vals.shape[0]):
        ax.plot(range(steps), f_vals[repeat, :steps], color="C0", linewidth=0.4)
    mean_vals = f_vals[:, :steps].mean(axis=0)
    std_vals = f_vals[:, :steps].std(axis=0)
    ax.plot(range(steps), mean_vals, color="C1", linewidth=1.5)
    ax.fill_between(
        range(steps),
        mean_vals - 2 * std_vals,
        mean_vals + 2 * std_vals,
        color="C1",
        alpha=0.5,
    )

    return f_vals[:, -1]


def boxplot_final_values(
    ax,
    cache: "SimulationCache",
    dims_list,
    repeats: int,
    steps: int,
    learning_rate: float = 1.0,
):
    """Ensure simulations for each dimension in `dims_list` are available
    and draw a boxplot of final values on `ax`.

    Returns the list of final-values arrays (one per dimension) used to build
    the boxplot."""
    final_values = []
    for dim in dims_list:
        cache.df = ensure_simulations(cache.df, dim, repeats, steps, learning_rate)
        subset = cache.df.filter(pl.col("dim") == dim).sort("seed")
        f_vals = np.stack(subset["f_vals"].to_list())
        final_values.append(f_vals[:, steps - 1])

    ax.boxplot(
        final_values,
        positions=dims_list,
        widths=0.5 * np.array(dims_list),
        manage_ticks=False,
    )
    ax.set_xlabel("dimension")
    ax.set_ylabel("function value")
    ax.set_xscale("log")
    ax.set_title(f"Distribution of values at step={steps}")
    return final_values


def plot_final_std(
    ax,
    cache: "SimulationCache",
    dims_list,
    repeats: int,
    steps: int,
    learning_rate: float = 1.0,
):
    """Plot the standard deviation of final values for each dimension on a log-log plot.

    Returns the array of std values.
    """
    stds = []
    for dim in dims_list:
        cache.df = ensure_simulations(cache.df, dim, repeats, steps, learning_rate)
        subset = cache.df.filter(pl.col("dim") == dim).sort("seed")
        f_vals = np.stack(subset["f_vals"].to_list())
        stds.append(np.std(f_vals[:, steps - 1]))

    ax.loglog(dims_list, stds, marker="o", color="C2", linewidth=1.5)
    ax.set_xlabel("dimension")
    ax.set_ylabel("STD of function values")
    ax.set_title(f"Standard deviation at step={steps} (log-log)")
    # theoretical 1/sqrt(dim) curve for comparison
    dims_arr = np.array(dims_list, dtype=float)
    theo = 1.0 / np.sqrt(dims_arr)
    ax.loglog(
        dims_arr,
        theo,
        linestyle="--",
        color="k",
        linewidth=1.0,
        label=r"$\frac{1}{\sqrt{\mathrm{dim}}}$",
    )
    ax.legend()
    return np.array(stds)


def plot_gd_overview(repeats=100, steps=20, learning_rate: float = 1.0):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    with SimulationCache("gd_simulations.parquet") as cache:
        axes = axes.flatten()
        for idx, dim in enumerate([1, 10, 100, 10_000]):
            plot_gd_trajectory(axes[idx], cache, dim, repeats, steps, learning_rate)

        # boxplot on the last subplot
        box_ax = axes[4]
        boxplot_final_values(
            box_ax,
            cache,
            [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000],
            repeats,
            steps,
            learning_rate,
        )

        # plot std of final values on the last subplot (axes[5])
        std_ax = axes[5]
        plot_final_std(
            std_ax,
            cache,
            [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 1_000_000],
            repeats,
            steps,
            learning_rate,
        )

        plt.tight_layout()
        plt.show()
    return fig


def dim_1_grf_plot():
    """Plot a 1D GRF sample."""
    f1 = IsotropicGRF(dim=1, kernel=SquaredExponentialKernel())
    x = np.arange(start=0, stop=10, step=0.1).reshape((-1, 1))
    y = f1(x)

    plt.plot(x.reshape(-1), y, color="C0", linewidth=0.8)
    plt.show()


def dim_2_grf_plot():
    """Plot a 2D GRF sample as a contour plot."""
    f2 = IsotropicGRF(dim=2, kernel=SquaredExponentialKernel())
    X, Y = np.mgrid[-5:5:0.4, -5:5:0.4]
    points = [np.array([x, y]) for x, y in zip(X.flatten(), Y.flatten())]
    z = np.array([f2(point) for point in points]).reshape(X.shape)
    plt.contour(X, Y, z, colors="C0", linewidths=0.5)
    plt.show()


if __name__ == "__main__":
    plot_gd_overview(repeats=100, steps=20, learning_rate=0.9)
    # dim_1_grf_plot()
    # dim_2_grf_plot()
    # simulate_gd(dim=1, steps=20)
