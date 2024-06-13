import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mosaic_scatter(cfg, results, metrics, title_str):
    def hexbin_1to1(ax, x, y, target, metrics):
        positive_mask = (x > 0) & (y > 0)
        x = x[positive_mask]
        y = y[positive_mask]
        
        min_val = 5E-3
        max_val = 5E6
        log_min = np.log10(5E-3)
        log_max = np.log10(5E6)

        hb = ax.hexbin(x, y, gridsize=(30,20), bins='log', mincnt=5, 
                    extent=(log_min, log_max, log_min, log_max),
                    xscale='log', yscale='log')
        plt.colorbar(hb, shrink=0.3, aspect=10, anchor=(-1,-0.55))
        
        # Add a 1:1 line over the min and max of x and y
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        textstr = '\n'.join([f"{key}: {metrics[target][key]:0.2f}" 
                             for key in ['nBias', 'RE', 'rRMSE', 'Agreement']])
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0, -0.4, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        # Setting axes to be square and equal range
        ax.axis('square')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_title(f"{target} (n {len(x)})")
        ax.set_xlabel(f'Observed')
        ax.set_ylabel(f'Predicted')

    targets = cfg['features']['target']
    fig, axes = plt.subplots(1,len(targets), figsize=(len(targets)*3, 4))

    axes = [axes] if len(targets)==1 else axes        
    for target, ax in zip(targets, axes):
        x = results['obs'][target]
        y = results['pred'][target]
        hexbin_1to1(ax, x, y, target, metrics)
        
    fig.subplots_adjust(top=0.9, bottom=0.3)
    fig.suptitle(title_str)

    return fig


def basin_metric_histograms(basin_metrics, metric_args):
    cols  = 3
    rows = int(np.ceil(len(metric_args)/cols))

    fig_dict = {}
    targets = basin_metrics.columns.get_level_values('Feature').unique()
    for target in targets:
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols,2*rows))
        axes = axes.flatten()

        for ax, (metric, plot_kwargs) in zip(axes, metric_args.items()):
            tmp = basin_metrics[target][metric].astype(float)
            valid_mask = (~np.isnan(tmp)) & (~np.isinf(tmp))
            tmp = tmp[valid_mask]

            ax.hist(tmp, bins=20, **plot_kwargs)
            # ax.scatter(basin_metrics[target]['num_obs'][valid_mask], tmp)
            ax.set_title(f"{metric} (m:{np.nanmedian(tmp):0.2f}, n:{np.sum(valid_mask)})")

        # Hide any unused axes.
        for ax in axes[len(metric_args):]:
            ax.set_axis_off()

        fig.suptitle(target.upper())
        fig.tight_layout()
        fig_dict[target] = fig
        
    return fig_dict