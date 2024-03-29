import numpy as np


def get_all_metrics(y, y_hat):
    metrics = {
        'mse': calc_mse(y, y_hat),
        'rmse': calc_rmse(y, y_hat),
        'kge': calc_kge(y, y_hat),
        'nse': calc_nse(y, y_hat)}
    return metrics

def calc_mse(y, y_hat):
    return np.nanmean((y - y_hat)**2)

def calc_rmse(y, y_hat):
    return np.sqrt(calc_mse(y, y_hat))

def calc_kge(y, y_hat):
    y, y_hat = _mask_nan(y, y_hat)
    
    # Calculate Pearson correlation coefficient
    correlation = np.corrcoef(y, y_hat)[0, 1]
    
    # Calculate mean and standard deviation
    mean_y = np.mean(y)
    mean_y_hat = np.mean(y_hat)
    std_y = np.std(y)
    std_y_hat = np.std(y_hat)
    
    # Calculate KGE
    kge_value = 1 - np.sqrt((correlation - 1)**2 + (std_y_hat/std_y - 1)**2 + (mean_y_hat/mean_y - 1)**2)
    
    return kge_value

def calc_nse(y, y_hat):
    y, y_hat = _mask_nan(y,y_hat)

    denominator = ((y_hat - y_hat.mean())**2).sum()
    numerator = ((y - y_hat)**2).sum()

    value = 1 - numerator / denominator

    return float(value)


def _mask_nan(y, y_hat):
    mask = (~np.isnan(y)) & (~np.isnan(y_hat))
    return y[mask], y_hat[mask]