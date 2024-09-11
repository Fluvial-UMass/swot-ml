import pandas as pd
import numpy as np
import sklearn.metrics as skm

log_pad = 0.001

def get_basin_metrics(df, disp=False):
    per_basin_metrics = df.groupby(level='basin').apply(get_all_metrics)

    # Initialize an empty DataFrame to store results with multi-level columns
    feature_names = df['obs'].columns
    metric_names = per_basin_metrics.iloc[0][feature_names[0]].keys()
    multi_index_columns = pd.MultiIndex.from_product([feature_names, metric_names],names=['Feature','Metric'])
    results_df = pd.DataFrame(columns=multi_index_columns)
    
    # Populate the DataFrame with the metrics results
    for basin, basin_metrics in per_basin_metrics.items():
        for feature, metrics in basin_metrics.items():
            for metric_name, metric_value in metrics.items():
                results_df.loc[basin, (feature, metric_name)] = metric_value

    if disp:
        metrics_str = "Basin Metrics\n"
        for feature in feature_names:
            metrics_str += f"{feature.upper()}\n"
            for metric in metric_names:
                metrics_str += f"{metric}: {np.nanmedian(results_df[feature][metric]):0.4f}\n"
            metrics_str += "\n"
        print(metrics_str)
   
    return results_df

def get_all_metrics(df:pd.DataFrame, disp=False):
    metrics = {}
    
    for feature in df['obs'].columns:
        y = df[('obs', feature)]
        y_hat = df[('pred', feature)]

        metrics[feature] = {
            'num_obs': np.sum(~np.isnan(y)),
            'R2': mask_nan(skm.r2_score)(y,y_hat),
            'MAPE': mask_nan(skm.mean_absolute_percentage_error)(y, y_hat),
            'nBias': calc_nbias(y, y_hat),
            'RE': calc_rel_err(y, y_hat),
            'MAE': calc_mae(y, y_hat),
            'RMSE': calc_rmse(y, y_hat),
            'rRMSE': calc_rrmse(y, y_hat),
            'KGE': calc_kge(y, y_hat),
            'NSE': calc_nse(y, y_hat),
            'Agreement': calc_agreement(y, y_hat)}
        
    if disp:
        metrics_str = "Bulk Metrics\n"
        for feature, feature_metrics in metrics.items():
            metrics_str += f"{feature.upper()}\n"
            for metric, value in feature_metrics.items():
                metrics_str += f"{metric}: {value:0.4f}\n"
            metrics_str += "\n"
        print(metrics_str)


    return metrics

def mask_nan(func):
    def wrapper(y, y_hat, *args, **kwargs):
        mask = (~np.isnan(y)) & (~np.isnan(y_hat))
        if np.sum(mask)>1:
            y_masked = y[mask]
            y_hat_masked = y_hat[mask]
            return func(y_masked, y_hat_masked, *args, **kwargs)
        else:
            return np.nan
    return wrapper

@mask_nan
def calc_nbias(y, y_hat):
    mean_y = np.mean(y)
    return np.nanmean((y - y_hat) / mean_y)

@mask_nan
def calc_mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))

@mask_nan
def calc_rel_err(y, y_hat):
    exponent = np.median(np.abs(np.log10(y_hat / y)))
    return (10 ** exponent) - 1

@mask_nan
def calc_rmse(y, y_hat):
    return np.sqrt(np.mean((y - y_hat)**2))
    
@mask_nan
def calc_rrmse(y, y_hat):
    rmse = calc_rmse(y, y_hat)
    mean_y_hat = np.mean(y_hat)

    if mean_y_hat == 0:
        return np.nan
    else:
        return rmse/mean_y_hat*100
 
@mask_nan
def calc_kge(y, y_hat):
    correlation = np.corrcoef(y, y_hat)[0, 1]
    mean_y = np.mean(y)
    mean_y_hat = np.mean(y_hat)
    std_y = np.std(y)
    std_y_hat = np.std(y_hat)

    if std_y == 0 or mean_y == 0:
        return np.nan
    else:
        return 1 - np.sqrt((correlation - 1)**2 + (std_y_hat/std_y - 1)**2 + (mean_y_hat/mean_y - 1)**2) 

@mask_nan
def calc_nse(y, y_hat):
    denominator = ((y_hat - y_hat.mean())**2).sum()
    numerator = ((y - y_hat)**2).sum()

    if denominator == 0:
        return np.nan
    else:
        return 1 - (numerator / denominator)

@mask_nan
def calc_lnse(y, y_hat):
    log_y = np.log(y)
    log_yhat = np.log(y_hat)
    return calc_nse(log_y, log_yhat)

@mask_nan
def calc_agreement(y, y_hat):
    """ https://www.nature.com/articles/srep19401 """
    corr = np.corrcoef(y, y_hat)[0,1]
    if corr >= 0:
        kappa = 0
    else:
        kappa = 2 * np.abs(np.mean((y-np.mean(y))*(y_hat-np.mean(y_hat))))

    numerator = np.mean((y - y_hat)**2)
    denominator = np.var(y) + np.var(y_hat) + (np.mean(y) - np.mean(y_hat))**2 + kappa

    if denominator == 0:
        return np.nan
    else:
        return 1 - (numerator / denominator)