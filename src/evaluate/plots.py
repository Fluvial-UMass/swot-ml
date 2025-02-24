import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mosaic_scatter(cfg: dict, results: pd.DataFrame, metrics: pd.DataFrame, title: str):

    def hexbin_1to1(ax: plt.Axes, x: pd.Series, y: pd.Series, target: str):
        positive_mask = (x > 0) & (y > 0)
        x = x[positive_mask]
        y = y[positive_mask]

        min_val = 5E-1
        max_val = 5E6
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)

        hb = ax.hexbin(x,
                       y,
                       gridsize=(30, 20),
                       bins='log',
                       mincnt=5,
                       linewidth=0.2,
                       extent=(log_min, log_max, log_min, log_max),
                       xscale='log',
                       yscale='log')
        plt.colorbar(hb, shrink=0.3, aspect=10, anchor=(-1, -0.55))

        # Add a 1:1 line over the min and max of x and y
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')

        textstr = '\n'.join([
            f"{key}: {metrics[target][key]:0.2f}"
            for key in ['R2', 'RE', 'MAPE', 'nBias']
        ])
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0,
                -0.4,
                textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=props)

        # Setting axes to be square and equal range
        ax.axis('square')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_title(f"{target} (n {len(x):,})")
        ax.set_xlabel(f'Observed')
        ax.set_ylabel(f'Predicted')

    targets = cfg['features']['target']
    fig, axes = plt.subplots(1, len(targets), figsize=(len(targets) * 3, 4))

    axes = [axes] if len(targets) == 1 else axes
    for target, ax in zip(targets, axes):
        x = results['obs'][target]
        y = results['pred'][target]
        hexbin_1to1(ax, x, y, target)

    fig.subplots_adjust(top=0.9, bottom=0.3)
    fig.suptitle(title)

    return fig


def basin_metric_histograms(basin_metrics: pd.DataFrame,
                            metric_args: dict | None = None,
                            cdf: bool = True):
    if metric_args is None:
        metric_args = {
            'R2': {
                'range': [-1, 1]
            },
            'RE': {
                'range': [0, 100]
            },
            'KGE': {
                'range': [-1, 1]
            }
        }

    cols = 3
    rows = int(np.ceil(len(metric_args) / cols))

    if cdf:
        common_args = {
            'bins': 500,
            'cumulative': True,
            'density': True,
            'histtype': 'step'
        }
    else:
        common_args = {'bins': 20}

    fig_dict = {}
    targets = basin_metrics.columns.get_level_values('Feature').unique()
    for target in targets:
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2 * rows))
        axes = axes.flatten()

        count_mask = basin_metrics[target]['num_obs'] > 1
        for ax, (metric, metric_kwargs) in zip(axes, metric_args.items()):
            tmp = basin_metrics[target][metric].astype(float)
            valid_mask = (~np.isnan(tmp)) & (~np.isinf(tmp)) & count_mask
            tmp = tmp[valid_mask]

            ax.hist(tmp, **common_args, **metric_kwargs)
            ax.set_title(
                f"{metric} (m:{np.nanmedian(tmp):0.2f}, n:{np.sum(valid_mask)})")

            lims = metric_kwargs.get('range')
            if lims:
                ax.set_xlim(lims[0], lims[1])
            ax.set_ylim([0, 1])

        # Hide any unused axes.
        for ax in axes[len(metric_args):]:
            ax.set_axis_off()

        fig.suptitle(target.upper())
        fig.tight_layout()
        fig_dict[target] = fig

    return fig_dict
