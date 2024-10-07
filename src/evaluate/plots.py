import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors



def mosaic_scatter(cfg, results, metrics, title_str):
    def hexbin_1to1(ax, x, y, target, metrics):
        positive_mask = (x > 0) & (y > 0)
        x = x[positive_mask]
        y = y[positive_mask]
        
        min_val = 5E-1
        max_val = 5E6
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)

        hb = ax.hexbin(x, y, gridsize=(30,20), bins='log', mincnt=5, 
                    extent=(log_min, log_max, log_min, log_max),
                    xscale='log', yscale='log')
        plt.colorbar(hb, shrink=0.3, aspect=10, anchor=(-1,-0.55))
        
        # Add a 1:1 line over the min and max of x and y
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        textstr = '\n'.join([f"{key}: {metrics[target][key]:0.2f}" 
                             for key in ['R2', 'RE', 'MAPE', 'nBias']])
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


def basin_metric_histograms(basin_metrics, metric_args, cdf=True):
    cols  = 3
    rows = int(np.ceil(len(metric_args)/cols))

    if cdf:
        common_args = {
            'bins': 500,
            'cumulative': True, 
            'density': True,
            'histtype': 'step'}
    else:
        common_args = {'bins':20}

    fig_dict = {}
    targets = basin_metrics.columns.get_level_values('Feature').unique()
    for target in targets:
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols,2*rows))
        axes = axes.flatten()

        count_mask = basin_metrics[target]['num_obs']>1
        for ax, (metric, metric_kwargs) in zip(axes, metric_args.items()):
            tmp = basin_metrics[target][metric].astype(float)
            valid_mask = (~np.isnan(tmp)) & (~np.isinf(tmp)) & count_mask
            tmp = tmp[valid_mask]

            ax.hist(tmp, **common_args, **metric_kwargs)
            ax.set_title(f"{metric} (m:{np.nanmedian(tmp):0.2f}, n:{np.sum(valid_mask)})")

            lims = metric_kwargs.get('range')
            if lims:
                ax.set_xlim(lims[0], lims[1])
            ax.set_ylim([0,1])

        # Hide any unused axes.
        for ax in axes[len(metric_args):]:
            ax.set_axis_off()

        fig.suptitle(target.upper())
        fig.tight_layout()
        fig_dict[target] = fig
        
    return fig_dict


def map_animation(cfg, model, dataset, target, cmap_label, period, lim, dt_alpha=1, log=True, denorm=True):
    from data import HydroDataLoader
    from evaluate import model_iterate
    import matplotlib.animation as animation

    wqp_locs = gpd.read_file("/work/pi_kandread_umass_edu/tss-ml/data/NA_WQP/metadata/wqp_sites.shp")
    wqp_locs = wqp_locs.set_index('LocationID')
    wqp_locs = wqp_locs.to_crs("EPSG:5070")

    dataset.inference_mode = True
    dataloader_kwargs = dataset.date_batching(date_range=period) # Return batches where each batch is 1 day
    cfg.update(dataloader_kwargs)
    dataloader = HydroDataLoader(cfg, dataset)

    test_basins = wqp_locs.loc[dataset.all_basins]
    test_basins['y_hat'] = 0.0 
    target_idx = np.where([t == target for t in dataset.target])[0]

    def draw_year_progress(ax, date):
        year_start = pd.Timestamp(date.year, 1, 1)
        year_end = pd.Timestamp(date.year, 12, 31)
        progress = (date - year_start).days / (year_end - year_start).days
        
        ax.add_patch(plt.Rectangle((0.1, 0.05), 0.8, 0.04, fill=False, transform=ax.transAxes))
        ax.add_patch(plt.Rectangle((0.1, 0.05), 0.8 * progress, 0.04, facecolor='black', transform=ax.transAxes))
        text_str = date.strftime('%Y-%m-%d')
        ax.text(0.1, 0.1, text_str, transform=ax.transAxes, fontsize=14, ha='left', va='bottom')

    # Create an empty plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    fig.set_tight_layout(True)

    if log:
        norm = colors.LogNorm(vmin=lim[0], vmax=lim[1])
    else:
        norm = colors.Normalize(vmin=lim[0], vmax=lim[1])

    # Add color bar
    sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
    sm._A = []  # Empty array for the scalar mappable
    cbar = plt.colorbar(sm, ax=ax, shrink=0.65, aspect=15, label=cmap_label)
    
    # Initialize animation frame
    plot = test_basins.plot(column='y_hat', cmap='inferno', norm=norm, ax=ax)

    # Data iterator wrapper for frames
    def frames():
        for data in model_iterate(model, dataloader, True, False, True):
            yield data

    # Updates the plot based on the data passed by frames()
    def update(frame):
        # Function signature is fixed so we have to unpack.
        basin, date, y_hat, dt = frame
        alpha = np.where(dt[:,1]==0, 1, dt_alpha)
        ax.clear()

        # Insert the data into our gdf and then plot.
        test_basins.loc[basin, 'y_hat'] = y_hat[:, target_idx]
        test_basins.plot(column='y_hat', cmap='inferno', linewidth=0, alpha=alpha, norm=norm, ax=ax)

        draw_year_progress(ax, pd.Timestamp(date[0]))
        ax.set_axis_off()
        return plot

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=frames, 
                                interval=1000/10, repeat=True, cache_frame_data=False)
    
    return ani

    # ani.save('ssc_animation.mp4', writer='ffmpeg', dpi=300)
    