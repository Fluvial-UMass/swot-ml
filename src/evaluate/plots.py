from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from config import Config


def mosaic_scatter(cfg: Config, results: pd.DataFrame, metrics: pd.DataFrame, title: str):
    def hexbin_1to1(ax: plt.Axes, x: pd.Series, y: pd.Series, target: str):
        positive_mask = (x > 0) & (y > 0)
        x = x[positive_mask]
        y = y[positive_mask]

        hb = ax.hexbin(
            x,
            y,
            gridsize=(30, 20),
            bins="log",
            mincnt=5,
            linewidth=0.2,
            xscale="log",
            yscale="log",
        )
        plt.colorbar(hb,shrink=0.3,aspect=10,anchor=(-1, -0.55))

        # Add a 1:1 line over the min and max of x and y
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot(xlim, ylim, "r--")

        textstr = "\n".join(
            [f"{key}: {metrics[target][key]:0.2f}" for key in ["R2", "RE", "MAPE", "nBias"]]
        )
        props = dict(boxstyle="round", facecolor="white", alpha=0.5)
        ax.text(0,-0.4,textstr,transform=ax.transAxes,fontsize=10,verticalalignment="top",bbox=props)

        # Setting axes to be square and equal range
        ax.axis("square")
        ax.set_xlim(min([xlim, ylim]))
        ax.set_ylim(max([xlim, ylim]))
        ax.set_title(f"{target} (n {len(x):,})")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")

    targets = cfg.features.target
    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(len(targets) * 3, 4),
    )

    axes = [axes] if len(targets) == 1 else axes
    for target, ax in zip(targets, axes):
        x = results["obs"][target]
        y = results["pred"][target]
        hexbin_1to1(ax, x, y, target)

    fig.subplots_adjust(top=0.9, bottom=0.3)
    fig.suptitle(title)

    return fig


def basin_metric_histograms(
    basin_metrics: pd.DataFrame,
    metric_args: dict | None = None,
    cdf: bool = True,
):
    if metric_args is None:
        metric_args = {
            "R2": {"range": [-1, 1]},
            "RE": {"range": [0, 100]},
            "KGE": {"range": [-1, 1]},
        }

    cols = 3
    rows = int(np.ceil(len(metric_args) / cols))

    if cdf:
        common_args = {
            "bins": 500,
            "cumulative": True,
            "density": True,
            "histtype": "step",
        }
    else:
        common_args = {"bins": 20}

    fig_dict = {}
    targets = basin_metrics.columns.get_level_values("Feature").unique()
    for target in targets:
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(3 * cols, 2 * rows),
        )
        axes = axes.flatten()

        count_mask = basin_metrics[target]["num_obs"] > 1
        for ax, (metric, metric_kwargs) in zip(axes, metric_args.items()):
            tmp = basin_metrics[target][metric].astype(float)
            valid_mask = (~np.isnan(tmp)) & (~np.isinf(tmp)) & count_mask
            tmp = tmp[valid_mask]

            ax.hist(
                tmp,
                **common_args,
                **metric_kwargs,
            )
            ax.set_title(f"{metric} (m:{np.nanmedian(tmp):0.2f}, n:{np.sum(valid_mask)})")

            lims = metric_kwargs.get("range")
            if lims:
                ax.set_xlim(lims[0], lims[1])
            ax.set_ylim([0, 1])

        # Hide any unused axes.
        for ax in axes[len(metric_args) :]:
            ax.set_axis_off()

        fig.suptitle(target.upper())
        fig.tight_layout()
        fig_dict[target] = fig

    return fig_dict


def plot_average_attribution(save_dir: Path, dataset):
    label_name_dict = {
        "snowmelt_sum": "Snowmelt",
        "Blue": "Blue",
        "Swir2": "SWIR 2",
        "Green": "Green",
        "surface_runoff_sum": "Surface runoff",
        "total_evaporation_sum": "Evaporation",
        "Swir1": "SWIR 1",
        "Red": "Red",
        "surface_solar_radiation_downwards_sum": "Shortwave radiation",
        "dw": "Dominant wavelength",
        "surface_thermal_radiation_downwards_sum": "Longwave radiation",
        "temperature_2m": "Air temperature (2m)",
        "pCount_dswe1": "Inundated pixel count",
        "Surface_temp_kelvin": "Surface temperature",
        "Nir": "NIR",
        "hue": "Hue",
        "total_precipitation_sum": "Precipitation",
    }

    targets = dataset.features["target"]
    feat_groups = list(dataset.features["dynamic"].keys())
    colors = list(mcolors.TABLEAU_COLORS.keys())

    # Create legend patches
    leg_patches = [mpatches.Patch(color=colors[i], label=g) for i, g in enumerate(feat_groups)]

    plt.close("all")
    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(4 * len(targets), 5),
    )
    axes = [axes] if len(targets) == 1 else axes

    for ax, target in zip(axes, targets):
        pkl_file = save_dir / f"{target}_integrated_gradients.pkl"
        with open(pkl_file, "rb") as file:
            ig = pickle.load(file)
        ig.replace(np.nan, 0, inplace=True)

        avg_imp = ig.mean(axis=0).abs()
        avg_imp.drop(
            [
                *dataset.d_encoding["encoded_columns"],
                target,
            ],
            inplace=True,
        )
        avg_imp = pd.DataFrame(avg_imp.rename("importance"))

        for k in avg_imp.index:
            for i, g in enumerate(feat_groups):
                if k in dataset.features["dynamic"][g]:
                    avg_imp.loc[k, "color"] = colors[i]
                    break

        avg_imp.rename(label_name_dict, inplace=True)
        avg_imp.sort_values(by="importance", inplace=True)

        ax.barh(avg_imp.index, avg_imp["importance"], color=avg_imp["color"])

        ax.legend(handles=leg_patches, loc="lower right")

        ax.set_title(f"{target}")
        ax.set_xlabel("Average IG")
        ax.ticklabel_format(
            style="sci",
            axis="x",
            useMathText=True,
            scilimits=(0, 0),
        )

    plt.tight_layout()
    fig.savefig(save_dir / "feature_attr.png", dpi=300)
