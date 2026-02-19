import numpy as np
import torch
import matplotlib.pyplot as plt


def two_frequency_convert_to_lowest(Y, X_M, d2):
    '''
    Input: 
        Y: np.ndarray of shape (P3, T)
        X_M: np.ndarray of shape (P2, T*d2)
        d2: int, number of medium frequency periods in one low frequency period
    Output:
        Y: np.ndarray of shape (P3 + d2*P2, T)
    '''
    P2 = X_M.shape[0]
    X_M_bar_reshape = reshape_last_dim_by_columns(X_M, d2)        #(P2, T, d2) (120, 100, 4)
    X_M_bar_reshape = X_M_bar_reshape.transpose(2, 0, 1).reshape(d2*P2, -1) #(d2*P2, T) (3* 40, 80)
    Y = np.concatenate((Y, X_M_bar_reshape), axis=0)        # (P3+d2*P2, T)
    return Y

def three_frequency_convert_to_lowest(Y, X_M, X_H, d1, d2):
    '''
    Input: 
        Y: np.ndarray of shape (P3, T)
        X_M: np.ndarray of shape (P2, T*d2)
        X_H: np.ndarray of shape (P1, T*d2*d1)
        d1: int, number of high frequency periods in one medium frequency period
        d2: int, number of medium frequency periods in one low frequency period
    Output:
        Y: np.ndarray of shape (P3 + d2*P2 + d1*d2*P1, T)
    '''
    P1 = X_H.shape[0]
    P2 = X_M.shape[0]
    X_H_reshape = reshape_last_dim_by_columns(X_H, d1)                #(P1, T*d2, d1) (40, 100*4, 3)
    X_H_reshape = X_H_reshape.transpose(2, 0, 1).reshape(d1 * P1, -1) #(d1* P1, T*d2) (3* 40, 100*4)
    X_M_bar = np.concatenate((X_M, X_H_reshape), axis=0)              # (P2+d1*P1, T*d2)
    X_M_bar_reshape = reshape_last_dim_by_columns(X_M_bar, d2)        #(P2+d1*P1, T, d2) (120, 100, 4)
    X_M_bar_reshape = X_M_bar_reshape.transpose(2, 0, 1).reshape(d1*d2*P1+d2*P2, -1) #(d1*d2*P1+d2*P2, T) (3* 40, 80)
    Y = np.concatenate((Y, X_M_bar_reshape), axis=0)        # (P3+d2*P2+d1*d2*P1, T)
    return Y

def reshape_last_dim_by_columns(data, num_cols):
    """
    Reshape the last dimension of a 2D array [P, x] into [P, x//num_cols, num_cols],
    filling values column-wise along the last dimension.

    Parameters:
        data (np.ndarray): Input array of shape [P, x]
        num_cols (int): Number of columns in the reshaped last dimension

    Returns:
        np.ndarray: Reshaped array of shape [P, x//num_cols, num_cols]
    """
    if len(data.shape) == 2:
        P, x = data.shape
        if x % num_cols != 0:
            raise ValueError("x must be divisible by num_cols")

        num_rows = x // num_cols
        reshaped = np.zeros((P, num_rows, num_cols), dtype=data.dtype)
        for col in range(num_cols):
            reshaped[:, :, col] = data[:, col::num_cols]
    elif len(data.shape) == 3:
        P, Q, x = data.shape
        if x % num_cols != 0:
            raise ValueError("x must be divisible by num_cols")

        new_num_rows = x // num_cols
        reshaped = np.zeros((P, Q, new_num_rows, num_cols), dtype=data.dtype)
        for col in range(num_cols):
            reshaped[:, :, :, col] = data[:, :, col::num_cols]

    return reshaped


def monthly_convert_to_quarterly(monthly_mat):
    P_month, n_month = monthly_mat.shape
    assert n_month % 3 == 0
    n_quarters = int(n_month/3)
    quarterly_mat = np.zeros((3*P_month, n_quarters))
    for i in range(P_month):
        for j in range(3):
            quarterly_mat[i*3 + j, :] = monthly_mat[i, j::3]
    return quarterly_mat

def quarterly_convert_to_monthly(quarterly_mat):
    '''
    Input:
        quarterly_mat: (3*P_month, n_quarters)
    '''
    P_quarter, n_quarters = quarterly_mat.shape
    assert P_quarter % 3 == 0
    P_month = int(P_quarter/3)
    # Calculate the number of months: 3 months per quarter
    n_months = n_quarters * 3
    # Initialize the original factors array
    monthly_mat = np.zeros((P_month, n_months))
    for i in range(P_month):
        for j in range(3):
            # Reconstruct the monthly series by placing quarterly values every 3 months
            monthly_mat[i, j::3] = quarterly_mat[i*3 + j, :]
    return monthly_mat


def create_lagged_data(data, lag):
    """
    Create lagged data from a 2D array of shape (D, T) into shape (D * lag, T + 1 - lag).

    Parameters:
        data (np.ndarray): Input array of shape (D, T)
        lag (int): Number of lag steps

    Returns:
        np.ndarray: Lagged array of shape (D * lag, T + 1 - lag)
    """
    D, T = data.shape
    if T < lag:
        raise ValueError("Time dimension T must be >= lag")

    lagged = []
    for i in range(lag):
        lagged.append(data[:, i:T - lag + i + 1])
    
    return np.vstack(lagged)

def create_lagged_data_torch(data: torch.Tensor, lag: int) -> torch.Tensor:
    """
    Create lagged data from a 2D tensor of shape (D, T) into shape (D * lag, T + 1 - lag).

    Parameters:
        data (torch.Tensor): Input tensor of shape (D, T)
        lag (int): Number of lag steps

    Returns:
        torch.Tensor: Lagged tensor of shape (D * lag, T + 1 - lag)
    """
    D, T = data.shape
    if T < lag:
        raise ValueError("Time dimension T must be >= lag")

    lagged = [data[:, i:T - lag + i + 1] for i in range(lag)]
    return torch.cat(lagged, dim=0)

def root_mean_squared_error(true, pred):
    squared_error = np.square(true - pred) 
    mse_t = np.mean(squared_error, axis=0) #suqare error for each quarter
    rmse_loss = np.sqrt(np.mean(mse_t))
    return mse_t, rmse_loss

def mean_absolute_error(true, pred):
    mae_t = np.mean(np.abs(true - pred), axis=0) # Mean absolute error for each quarter
    mae = np.mean(mae_t)  # Mean absolute error across all quarters
    return mae_t, mae

def standardize_per_variable(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize each row independently → returns (z, means, stds)"""
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    std_safe = np.where(std == 0, 1.0, std)     # or std[std==0] = 1.0
    z = (arr - mean) / std_safe
    return z, mean, std                         # keep original std (not safe version)

def destandardize_per_variable(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Reverse standardization using saved mean & std per sample"""
    mean = np.asarray(mean).reshape(-1, 1)
    std  = np.asarray(std).reshape(-1, 1)
    return z * std + mean

def normalize01_per_variable(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize to [0,1] per sample → returns (normalized, mins, maxs)"""
    mins = arr.min(axis=1, keepdims=True)
    maxs = arr.max(axis=1, keepdims=True)
    denom = maxs - mins
    norm = (arr - mins) / np.where(denom == 0, 1.0, denom)
    # Optional: set constant rows exactly to 0
    # norm[denom == 0] = 0.0
    return norm, mins, maxs

def denormalize01_per_variable(
    norm: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray
) -> np.ndarray:
    mins = np.asarray(mins).reshape(-1, 1)
    maxs = np.asarray(maxs).reshape(-1, 1)
    return norm * (maxs - mins) + mins


def print_stats(arr, name = 'Summary'):
    print(f"--- {name} ---")
    print('Shape:', arr.shape)
    print("Stat: max=%.4f  min=%.4f  mean=%.4f  std=%.4f"
          % (arr.max(), arr.min(), arr.mean(), arr.std()))



#############################plotting functions#############################
def plot_list(list_value):
    """
    Plots a list of error values.

    Parameters:
    list_value (list of float): The error values to plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(list_value, marker='o', linestyle='-', color='red')
    plt.title('Error List Over Time')
    plt.xlabel('Index')
    plt.ylabel('Error Value')
    plt.grid(True)
    plt.show()


def plot_forecasts_multi(true_array,
                   pred_array,
                   indices=None,
                   n_cols=3,
                   figsize=None,
                   title_prefix="Response variable",
                   save_path=None):
    """
    Plot true vs predicted values for selected (or all) response variables.
    
    Parameters
    ----------
    true_array : np.ndarray
        Shape (Q1, n_test) – ground truth
    pred_array : np.ndarray
        Shape (Q1, n_test) – model predictions
    indices : list or None
        List of indices (0 to Q1-1) you want to plot.
        If None → plot the first min(20, Q1) series.
    n_cols : int
        Number of columns in the subplot grid
    figsize : tuple or None
        Figure size. Auto-scaled if None.
    title_prefix : str
        Prefix for each subplot title
    save_path : str or None
        If provided, saves the figure (e.g. "forecasts.png")
    """
    Q1, n_test = true_array.shape
    title_list = ["GDPC1", "PCECC96", "GPDIC1", "GCEC1", "IPDBS"]
    
    if indices is None:
        indices = list(range(min(20, Q1)))
    else:
        indices = [i for i in indices if i < Q1]  # safety
    n_plots = len(indices)
    if n_plots == 0:

        print("No valid indices to plot.")
        return
    
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (6 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Pre-compute per-series metrics for titles
    mae_per_series  = np.mean(np.abs(true_array - pred_array), axis=1)
    rmse_per_series = np.sqrt(np.mean((true_array - pred_array)**2, axis=1))
    
    for idx, ax_idx in zip(indices, range(n_plots)):
        ax = axes[ax_idx]
        t = np.arange(n_test)
        
        ax.plot(t, true_array[idx], label='True', color='tab:blue', linewidth=1.8)
        ax.plot(t, pred_array[idx], label='Predicted', color='tab:orange', linewidth=1.8, alpha=0.9)
        
        ax.set_title(f"{title_list[idx]}\n"
                     f"MAE = {mae_per_series[idx]:.5f} | RMSE = {rmse_per_series[idx]:.5f}",
                     fontsize=11)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend()
    
    # Hide unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.xlabel("Test time step (one-step-ahead)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_forecasts(Y_total_true, Y_hat, T_train):
    """
    Plots P line plots comparing true and forecasted values.

    Parameters:
    - Y_true: numpy array of shape [P, T_train + T_test]
    - Y_hat: numpy array of shape [P, T_test]
    - T_train: int, length of training period

    Returns:
    - None (displays P plots)
    """
    import numpy as np

    P, total_time = Y_total_true.shape
    T_test = Y_hat.shape[1]

    for i in range(P):
        plt.figure(figsize=(10, 4))
        # Plot true values
        plt.plot(range(total_time), Y_total_true[i], label='True', color='blue')
        # Plot forecasted values
        plt.plot(range(T_train, T_train + T_test), Y_hat[i], label='Forecast', color='red')
        plt.title(f'Series {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_forecasts_221(Y_total_true, Y_hat, T_train):
    """
    Plots 5 line plots in a 2+2+1 layout comparing true and forecasted values.

    Parameters:
    - Y_total_true: numpy array of shape [P, T_train + T_test]
    - Y_hat: numpy array of shape [P, T_test]
    - T_train: int, length of training period

    Returns:
    - None (displays a single figure with 5 subplots)
    """
    P = 5
    T_test = Y_hat.shape[1]

    fig = plt.figure(figsize=(16, 10))

    # Top row: 2 plots
    for i in range(2):
        ax = fig.add_subplot(3, 2, i + 1)
        ax.plot(range(Y_total_true.shape[1]), Y_total_true[i], label='True', color='blue')
        ax.plot(range(T_train, T_train + T_test), Y_hat[i], label='Forecast', color='red')
        ax.set_title(f'Series {i+1}')
        ax.grid(True)

    # Middle row: 2 plots
    for i in range(2, 4):
        ax = fig.add_subplot(3, 2, i + 1)
        ax.plot(range(Y_total_true.shape[1]), Y_total_true[i], label='True', color='blue')
        ax.plot(range(T_train, T_train + T_test), Y_hat[i], label='Forecast', color='red')
        ax.set_title(f'Series {i+1}')
        ax.grid(True)

    # Bottom row: 1 plot centered
    ax = fig.add_subplot(3, 2, 5)
    ax.plot(range(Y_total_true.shape[1]), Y_total_true[4], label='True', color='blue')
    ax.plot(range(T_train, T_train + T_test), Y_hat[4], label='Forecast', color='red')
    ax.set_title('Series 5')
    ax.grid(True)

    # Hide the empty subplot (position 6)
    fig.add_subplot(3, 2, 6).axis('off')

    # Add legend to the first subplot only
    fig.axes[0].legend(loc='upper right')

    plt.tight_layout()
    plt.show()
