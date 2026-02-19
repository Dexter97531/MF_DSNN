import os
import sys
sys.path.append(os.path.abspath(os.path.join('/home/yzwang/MF')))
import random
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import (reshape_last_dim_by_columns, create_lagged_data)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)           # multi-GPU
    # torch.use_deterministic_algorithms(True)   # very important in recent PyTorch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
################################ Ours ################################
# Define NN for medium frequency factors
class DNN3_G1(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, W=12, L=2):
        super(DNN3_G1, self).__init__()
        self.L = L
        self.W = W
        self.fc1 = nn.Linear(input_dim, W)
        self.fc2 = nn.Linear(W, W)
        self.fc3 = nn.Linear(W, W)
        self.fc4 = nn.Linear(W, W)
        self.fcL = nn.Linear(W, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        if self.L > 1:
            x = self.relu(self.fc2(x))
        if self.L > 2:
            x = self.relu(self.fc3(x))
        if self.L > 3:
            x = self.relu(self.fc4(x))
        x = self.fcL(x)
        return x

# Define NN for low frequency factors
class DNN3_G2(nn.Module):
    def __init__(self, input_dim=4, output_dim=2, W=12, L=2):
        super(DNN3_G2, self).__init__()
        self.L = L
        self.W = W
        self.fc1 = nn.Linear(input_dim, W)
        self.fc2 = nn.Linear(W, W)
        self.fc3 = nn.Linear(W, W)
        self.fc4 = nn.Linear(W, W)
        self.fcL = nn.Linear(W, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        if self.L > 1:
            x = self.relu(self.fc2(x))
        if self.L > 2:
            x = self.relu(self.fc3(x))
        if self.L > 3:
            x = self.relu(self.fc4(x))
        x = self.fcL(x)
        return x


class DNN3_G3(nn.Module):
    def __init__(self, input_dim=735, output_dim=5, W=2048, dropout_rate=0.3, L=4):
        super(DNN3_G3, self).__init__()
        self.L = L
        self.fc1 = nn.Linear(input_dim, W)
        self.fc2 = nn.Linear(W, W)
        self.fc3 = nn.Linear(W, W)
        self.fc4 = nn.Linear(W, W)
        self.fc5 = nn.Linear(W, W)
        self.fcL = nn.Linear(W, output_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)  # Single dropout layer (shared)
    
    def forward(self, x):
        # x = torch.cat((exog_monthly, exog_quarterly), dim=1)
        x = self.relu(self.fc1(x))
        if self.L > 1:
            x = self.dropout(x)  # Dropout after first hidden layer
            x = self.relu(self.fc2(x))
        if self.L > 2:
            x = self.dropout(x)  # Dropout after second
            x = self.relu(self.fc3(x))
        if self.L > 3:
            x = self.dropout(x)  # Dropout after third
            x = self.relu(self.fc4(x))
        if self.L > 4:
            x = self.dropout(x)  # Dropout after third
            x = self.relu(self.fc5(x))
        # No dropout before output (common practice)
        x = self.fcL(x)
        return x


def rolling_forward_forecast_output_batch_v1(
    Y_train, X_M_train, X_H_train, Y_test, X_M_test, X_H_test,
    d1=20, d2=3, r1=1, r2=2, k=2, L=2, L3=4, epochs=100, scaler=None,
    hidden_dim=2048, hidden_dim1 = 12, hidden_dim2 = 12,
    batch_size=32, lr=1e-4, Q=5, device='cpu',
    print_importance=False
):
    set_seed(42)
    P1, T_train_full = X_H_train.shape
    P2, _ = X_M_train.shape
    QP3, _ = Y_train.shape
    T_train = T_train_full // (d1 * d2)
    T_test = Y_test.shape[1]

    D2 = P2 + r1 * P1
    D = r1 * r2 * P1 + r2 * P2 + QP3
    forecast_test = np.zeros((Q, T_test))
    forecast_true = np.zeros((Q, T_test))

    for t in range(T_test):
        # print(f'==== Step {t} ====')
        # Move models to device
        device = torch.device(device)
        # === Initialize models ===
        DNN_g1 = DNN3_G1(input_dim=d1, output_dim=r1, W=hidden_dim1, L=L).to(device)
        DNN_g2 = DNN3_G2(input_dim=d2, output_dim=r2, W=hidden_dim2, L=L).to(device)
        DNN2_g3 = DNN3_G3(input_dim=D*k, output_dim=Q, W=hidden_dim, L=L3).to(device)

        optimizer = optim.Adam(
            list(DNN_g1.parameters()) + list(DNN_g2.parameters()) + list(DNN2_g3.parameters()),
            lr=lr, weight_decay=1e-4
        )
        criterion = nn.MSELoss()
        # === Update rolling window data ===
        if t == 0:
            X_H_train_cur = X_H_train[:, :-d1*d2]
            X_M_train_cur = X_M_train[:, :-d2]
            X_L_train_cur = Y_train[:, :-1]
            Y_train_cur = Y_train[:Q, k:]
        elif t == 1:
            X_H_train_cur = np.hstack((X_H_train_cur, X_H_train[:, -d1*d2:]))
            X_M_train_cur = np.hstack((X_M_train_cur, X_M_train[:, -d2:]))
            X_L_train_cur = np.hstack((X_L_train_cur, Y_train[:, -1].reshape(-1,1)))
            Y_train_cur = np.hstack((Y_train_cur, Y_test[:Q, t-1].reshape(-1,1)))
        else:
            X_H_train_cur = np.hstack((X_H_train_cur, X_H_test[:, (t-2)*d1*d2:(t-1)*d1*d2]))
            X_M_train_cur = np.hstack((X_M_train_cur, X_M_test[:, (t-2)*d2:(t-1)*d2]))
            X_L_train_cur = np.hstack((X_L_train_cur, Y_test[:, t-2].reshape(-1,1)))
            Y_train_cur = np.hstack((Y_train_cur, Y_test[:Q, t-1].reshape(-1,1)))

        T_cur = Y_train_cur.shape[1]

        train_threshold = 1e-3
        error_epoch = []

        DNN_g1.train()
        DNN_g2.train()
        DNN2_g3.train()

        # === Training Loop with Batches ===
        for epoch in range(epochs):
            epoch_loss = 0.0
            # Reshape high-frequency data: (P1, T_cur*d2, d1)
            X_H_reshape = reshape_last_dim_by_columns(X_H_train_cur, d1)  # (P1, T_cur*d2, d1)

            # We'll collect factor sequences across time
            F_M_list = []
            for p in range(P1):
                x_h_p = torch.FloatTensor(X_H_reshape[p]).to(device)  # (T_cur*d2, d1)
                f_m_p = DNN_g1(x_h_p).T  # (r1, T_cur*d2)
                F_M_list.append(f_m_p)
            F_M = torch.cat(F_M_list, dim=0)  # (P1*r1, T_cur*d2)

            X_M_bar = np.vstack((X_M_train_cur, F_M.detach().cpu().numpy()))  # (D2, T_cur*d2)
            X_M_bar_reshape = reshape_last_dim_by_columns(X_M_bar, d2)  # (D2, T_cur, d2)

            F_L_list = []
            for p in range(D2):
                x_m_p = torch.FloatTensor(X_M_bar_reshape[p]).to(device)  # (T_cur, d2)
                f_l_p = DNN_g2(x_m_p).T  # (r2, T_cur)
                F_L_list.append(f_l_p)
            F_L = torch.cat(F_L_list, dim=0)  # (D2*r2, T_cur)

            X_L_bar = np.vstack((X_L_train_cur, F_L.detach().cpu().numpy()))  # (D, T_cur)
            X_lagged = create_lagged_data(X_L_bar, k).T  # (T_cur - k + 1, D*k)
            y_target = Y_train_cur  # (P3, T_cur - k + 1)
            # print('X_lagged',X_lagged.shape)
            # print('y_target',y_target.shape)
            # Ensure lengths match
            assert X_lagged.shape[0] == y_target.shape[1], f"X_lagged {X_lagged.shape}, y {y_target.shape}"

            # Create dataset and loader
            dataset = TensorDataset(
                torch.FloatTensor(X_lagged).to(device),
                torch.FloatTensor(y_target.T).to(device)  # (T-k+1, P3)
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for x_batch, y_batch in loader:
                optimizer.zero_grad()

                # Forward
                y_pred = DNN2_g3(x_batch)  # (B, P3)
                loss = criterion(y_pred, y_batch)

                # Backward
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item() * x_batch.size(0)

            epoch_loss /= len(loader.dataset)
            error_epoch.append(epoch_loss)

            if t < 1 and epoch % 300 == 0:
                print(f'Step {t}, Epoch {epoch}, Train Loss: {epoch_loss:.6f}')

            if epoch_loss < train_threshold:
                print(f'Loss {epoch_loss:.6f} < {train_threshold} at epoch {epoch}, early stop.')
                break

        # === Validation / One-step Forecast ===
        
        if t == 0:
            # plot_list(error_epoch)
            X_H_test_cur = X_H_train[:, -k*d1*d2:]
            X_M_test_cur = X_M_train[:, -k*d2:]
            X_L_test_cur = Y_train[:, -k:]
        else:
            
            X_H_test_cur = np.hstack((X_H_test_cur[:, d1*d2:], X_H_test[:, (t-1)*d1*d2 : t*d1*d2]))
            X_M_test_cur = np.hstack((X_M_test_cur[:, d2:], X_M_test[:, (t-1)*d2 : t*d2]))
            X_L_test_cur = np.hstack((X_L_test_cur[:, 1:], Y_test[:, t-1].reshape(-1,1)))

        Y_test_cur = Y_test[:Q, t].reshape(-1, 1)
    
        DNN_g1.eval()
        DNN_g2.eval()
        DNN2_g3.eval()
        with torch.no_grad():
            # High → Medium factors
            X_H_test_reshape = reshape_last_dim_by_columns(X_H_test_cur, d1)  # (P1, k*d2, d1)
            F_M_list = [DNN_g1(torch.FloatTensor(X_H_test_reshape[p]).to(device)).T for p in range(P1)]
            F_M = torch.cat(F_M_list, dim=0)  # (P1*r1, k*d2)

            X_M_bar_test = np.vstack((X_M_test_cur, F_M.cpu().numpy()))  # (D2, k*d2)
            X_M_bar_test_reshape = reshape_last_dim_by_columns(X_M_bar_test, d2)  # (D2, k, d2)

            F_L_list = [DNN_g2(torch.FloatTensor(X_M_bar_test_reshape[p]).to(device)).T for p in range(D2)]
            F_L = torch.cat(F_L_list, dim=0)  # (D2*r2, k)

            X_L_bar_test = np.vstack((X_L_test_cur, F_L.cpu().numpy()))  # (D, k)
            X_lagged_test = create_lagged_data(X_L_bar_test, k).T  # (1, D*k)

            y_pred = DNN2_g3(torch.FloatTensor(X_lagged_test).to(device)).cpu().numpy().flatten()

        forecast_test[:, t] = y_pred
        forecast_true[:, t] = Y_test_cur.flatten()
    if print_importance:
        print_model_importance(
            DNN_g1=DNN_g1,
            DNN_g2=DNN_g2,
            DNN2_g3=DNN2_g3,
            X_H_train_cur=X_H_train_cur,
            X_M_train_cur=X_M_train_cur,
            Y_train_cur=Y_train_cur,       # not directly used, but kept for consistency
            d1=d1,
            d2=d2,
            r1=r1,
            r2=r2,
            k=k,
            P1=P1,
            P2=P2,
            QP3=QP3,
            device=device
        )

    return forecast_test, forecast_true


def rolling_forward_forecast_output_batch_v2(
    Y_train, X_M_train, X_H_train, Y_test, X_M_test, X_H_test,
    d1=20, d2=3, r1=1, r2=2, L=2, L3=4, k=2, epochs=100, scaler=None,
    hidden_dim=2048, hidden_dim1 = 12, hidden_dim2 = 12,
    batch_size=32, lr=1e-4, Q=5, device='cpu',
    print_importance=False, decay = False, early_stopping = False
):
    set_seed(42)
    P1, T_train_full = X_H_train.shape
    P2, _ = X_M_train.shape
    QP3, _ = Y_train.shape
    T_train = T_train_full // (d1 * d2)
    T_test = Y_test.shape[1]

    D2 = P2 + r1 * P1
    D = r1 * r2 * P1 + r2 * P2 + QP3
    forecast_test = np.zeros((Q, T_test))
    forecast_true = np.zeros((Q, T_test))
    device = torch.device(device)
    # === Initialize models ===
    DNN_g1 = DNN3_G1(input_dim=d1, output_dim=r1, W=hidden_dim1, L=L).to(device)
    DNN_g2 = DNN3_G2(input_dim=d2, output_dim=r2, W=hidden_dim2, L=L).to(device)
    DNN2_g3 = DNN3_G3(input_dim=D*k, output_dim=Q, W=hidden_dim, L=L3).to(device)

    optimizer = optim.Adam(
        list(DNN_g1.parameters()) + list(DNN_g2.parameters()) + list(DNN2_g3.parameters()),
        lr=lr, weight_decay=1e-4
    )
    criterion = nn.MSELoss()

    for t in range(T_test):
        # === Update rolling window data ===
        if t == 0:
            X_H_train_cur = X_H_train[:, :-d1*d2]
            X_M_train_cur = X_M_train[:, :-d2]
            X_L_train_cur = Y_train[:, :-1]
            Y_train_cur = Y_train[:Q, k:]
        elif t == 1:
            X_H_train_cur = np.hstack((X_H_train_cur, X_H_train[:, -d1*d2:]))
            X_M_train_cur = np.hstack((X_M_train_cur, X_M_train[:, -d2:]))
            X_L_train_cur = np.hstack((X_L_train_cur, Y_train[:, -1].reshape(-1,1)))
            Y_train_cur = np.hstack((Y_train_cur, Y_test[:Q, t-1].reshape(-1,1)))
        else:
            X_H_train_cur = np.hstack((X_H_train_cur, X_H_test[:, (t-2)*d1*d2:(t-1)*d1*d2]))
            X_M_train_cur = np.hstack((X_M_train_cur, X_M_test[:, (t-2)*d2:(t-1)*d2]))
            X_L_train_cur = np.hstack((X_L_train_cur, Y_test[:, t-2].reshape(-1,1)))
            Y_train_cur = np.hstack((Y_train_cur, Y_test[:Q, t-1].reshape(-1,1)))

        T_cur = Y_train_cur.shape[1]

        train_threshold = 1e-3
        change_percent = np.inf
        error_epoch = []

        DNN_g1.train()
        DNN_g2.train()
        DNN2_g3.train()

        # === Training Loop with Batches ===
        for epoch in range(epochs):
            epoch_loss = 0.0
            # Reshape high-frequency data: (P1, T_cur*d2, d1)
            X_H_reshape = reshape_last_dim_by_columns(X_H_train_cur, d1)  # (P1, T_cur*d2, d1)

            # We'll collect factor sequences across time
            F_M_list = []
            for p in range(P1):
                x_h_p = torch.FloatTensor(X_H_reshape[p]).to(device)  # (T_cur*d2, d1)
                f_m_p = DNN_g1(x_h_p).T  # (r1, T_cur*d2)
                F_M_list.append(f_m_p)
            F_M = torch.cat(F_M_list, dim=0)  # (P1*r1, T_cur*d2)

            X_M_bar = np.vstack((X_M_train_cur, F_M.detach().cpu().numpy()))  # (D2, T_cur*d2)
            X_M_bar_reshape = reshape_last_dim_by_columns(X_M_bar, d2)  # (D2, T_cur, d2)

            F_L_list = []
            for p in range(D2):
                x_m_p = torch.FloatTensor(X_M_bar_reshape[p]).to(device)  # (T_cur, d2)
                f_l_p = DNN_g2(x_m_p).T  # (r2, T_cur)
                F_L_list.append(f_l_p)
            F_L = torch.cat(F_L_list, dim=0)  # (D2*r2, T_cur)

            X_L_bar = np.vstack((X_L_train_cur, F_L.detach().cpu().numpy()))  # (D, T_cur)
            X_lagged = create_lagged_data(X_L_bar, k).T  # (T_cur - k + 1, D*k)
            y_target = Y_train_cur  # (P3, T_cur - k + 1)
            # print('X_lagged',X_lagged.shape)
            # print('y_target',y_target.shape)
            # Ensure lengths match
            assert X_lagged.shape[0] == y_target.shape[1], f"X_lagged {X_lagged.shape}, y {y_target.shape}"

            # Create dataset and loader
            dataset = TensorDataset(
                torch.FloatTensor(X_lagged).to(device),
                torch.FloatTensor(y_target.T).to(device)  # (T-k+1, P3)
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for x_batch, y_batch in loader:
                optimizer.zero_grad()

                # Forward
                y_pred = DNN2_g3(x_batch)  # (B, P3)
                loss = criterion(y_pred, y_batch)

                # Backward
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item() * x_batch.size(0)

            epoch_loss /= len(loader.dataset)
            error_epoch.append(epoch_loss)
            change_percent = np.abs((error_epoch[-2] - epoch_loss) / (error_epoch[-2] + 1e-12)) if epoch >=1 else 0.0

            if t < 1 and epoch % 90 == 0:
                print(f'Step {t}, Epoch {epoch}, Train Loss: {epoch_loss:.6f}')

            if epoch_loss < train_threshold:
                print(f'Loss {epoch_loss:.6f} < {train_threshold} at Step {t}, epoch {epoch}, early stop.')
                break
            if early_stopping and epoch > 10 and change_percent < 1e-5:
                print(f'change_percent < 1e-5 at Step {t}, epoch {epoch}, early stop.')
                break

        # === Validation / One-step Forecast ===
        if t == 0:
            X_H_test_cur = X_H_train[:, -k*d1*d2:]
            X_M_test_cur = X_M_train[:, -k*d2:]
            X_L_test_cur = Y_train[:, -k:]
        else:
            X_H_test_cur = np.hstack((X_H_test_cur[:, d1*d2:], X_H_test[:, (t-1)*d1*d2 : t*d1*d2]))
            X_M_test_cur = np.hstack((X_M_test_cur[:, d2:], X_M_test[:, (t-1)*d2 : t*d2]))
            X_L_test_cur = np.hstack((X_L_test_cur[:, 1:], Y_test[:, t-1].reshape(-1,1)))
        
        Y_test_cur = Y_test[:Q, t].reshape(-1, 1)
    
        DNN_g1.eval()
        DNN_g2.eval()
        DNN2_g3.eval()
        with torch.no_grad():
            # High → Medium factors
            X_H_test_reshape = reshape_last_dim_by_columns(X_H_test_cur, d1)  # (P1, k*d2, d1)
            F_M_list = [DNN_g1(torch.FloatTensor(X_H_test_reshape[p]).to(device)).T for p in range(P1)]
            F_M = torch.cat(F_M_list, dim=0)  # (P1*r1, k*d2)

            X_M_bar_test = np.vstack((X_M_test_cur, F_M.cpu().numpy()))  # (D2, k*d2)
            X_M_bar_test_reshape = reshape_last_dim_by_columns(X_M_bar_test, d2)  # (D2, k, d2)

            F_L_list = [DNN_g2(torch.FloatTensor(X_M_bar_test_reshape[p]).to(device)).T for p in range(D2)]
            F_L = torch.cat(F_L_list, dim=0)  # (D2*r2, k)

            X_L_bar_test = np.vstack((X_L_test_cur, F_L.cpu().numpy()))  # (D, k)
            X_lagged_test = create_lagged_data(X_L_bar_test, k).T  # (1, D*k)

            y_pred = DNN2_g3(torch.FloatTensor(X_lagged_test).to(device)).cpu().numpy().flatten()

        forecast_test[:, t] = y_pred
        forecast_true[:, t] = Y_test_cur.flatten()
    if print_importance:
        print_model_importance(
            DNN_g1=DNN_g1, DNN_g2=DNN_g2, DNN2_g3=DNN2_g3,
            X_H_train_cur=X_H_train_cur, X_M_train_cur=X_M_train_cur,
            Y_train_cur=Y_train_cur,       # not directly used, but kept for consistency
            d1=d1, d2=d2, r1=r1, r2=r2, k=k,
            P1=P1, P2=P2, QP3=QP3,
            device=device
        )
    return forecast_test, forecast_true


def print_spectral_norms_respecting_L(model, model_name="Model", decimals=6):
    """
    Prints spectral norm (max operator norm) only for the active layers:
    fc1 → fcL (according to model.L) + the final fcL layer.
    Total: model.L + 1 linear layers.
    """
    L = model.L  # the number of "intermediate" layers before the last one
    
    print(f"\n=== {model_name} (L={L}) — Spectral norms of active weight matrices ===")
    print(f"{'Layer':<10} {'Shape':<16} {'Max op norm (spectral)':>18}  ")
    print("-" * 60)
    
    # fc1 is always present
    layers_to_check = ["fc1"]
    
    # Add fc2 to fc{L} if L >= 2,3,...
    for i in range(2, L+1):
        layers_to_check.append(f"fc{i}")
    
    # Always include the final layer (usually called fcL)
    layers_to_check.append("fcL")
    
    for layer_name in layers_to_check:
        if not hasattr(model, layer_name):
            continue
            
        linear_layer = getattr(model, layer_name)
        if not isinstance(linear_layer, torch.nn.Linear):
            continue
            
        weight = linear_layer.weight.data
        if weight.ndim != 2:
            continue
            
        singular_values = torch.linalg.svdvals(weight)
        max_norm = singular_values[0].item()
        
        shape_str = f"({weight.shape[0]} × {weight.shape[1]})"
        print(f"{layer_name:<10} {shape_str:<16} {max_norm:>18.{decimals}f}")


def print_model_importance(
    DNN_g1, DNN_g2, DNN2_g3,
    X_H_train_cur, X_M_train_cur, Y_train_cur,
    d1, d2, r1, r2, k,
    P1, P2, QP3,
    device='cpu'
):
    """
    Comprehensive importance analysis for the three-stage factor model.
    
    Prints:
      - Weight-based and Gradient-based importance for DNN_g1 inputs (d1 lags)
      - Weight-based and Gradient-based importance for DNN_g2 inputs (d2 lags)
      - Original feature importance from DNN2_g3 (via existing function)
    """
    device = next(DNN_g1.parameters()).device  # ensure correct device

    # ======================================================
    # Prepare data for g1 and g2 (same as in training loop)
    # ======================================================
    X_H_reshape = reshape_last_dim_by_columns(X_H_train_cur, d1)  # (P1, T*d2, d1)

    # Compute F_M (high → medium factors)
    with torch.no_grad():
        F_M_list = []
        for p in range(P1):
            x_h_p = torch.FloatTensor(X_H_reshape[p]).to(device)
            f_m_p = DNN_g1(x_h_p).T  # (r1, T*d2)
            F_M_list.append(f_m_p)
        F_M = torch.cat(F_M_list, dim=0)  # (P1*r1, T*d2)

        X_M_bar = np.vstack((X_M_train_cur, F_M.cpu().numpy()))  # (D2, T*d2)
        X_M_bar_reshape = reshape_last_dim_by_columns(X_M_bar, d2)  # (D2, T, d2)

    D2 = P2 + r1 * P1

    # ==================== Helper to get first linear weight ====================
    def get_first_linear_weight(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                return module.weight
        return None

    # ==================== DNN_g1 IMPORTANCE ====================
    print("\n" + "="*70)
    print("INPUT IMPORTANCE: DNN_g1 (High-frequency → Medium factors)")
    print(f"d1 = {d1} lags (from lag -{d1-1} to lag 0)")
    print("="*70)

    # Weight-based
    w_g1 = get_first_linear_weight(DNN_g1)
    if w_g1 is not None:
        weight_imp_g1 = torch.abs(w_g1).mean(dim=0).cpu().detach().numpy()
        weight_imp_g1 /= (weight_imp_g1.sum() + 1e-12)
    else:
        weight_imp_g1 = np.zeros(d1)
        print("Warning: No Linear layer in DNN_g1 → weight importance unavailable")

    # Gradient-based
    DNN_g1.eval()
    grad_imp_g1 = np.zeros(d1)
    count = 0
    for p in range(P1):
        x = torch.FloatTensor(X_H_reshape[p]).to(device)
        x.requires_grad_(True)
        out = DNN_g1(x)                  # (T*d2, r1)
        out.abs().sum().backward()
        grad_imp_g1 += x.grad.abs().mean(dim=0).cpu().numpy()
        count += 1
        x.grad.zero_()
    grad_imp_g1 /= max(count, 1)
    grad_imp_g1 /= (grad_imp_g1.sum() + 1e-12)

    # Print table
    print(f"{'Lag':>5} | {'Weight-based':>16} | {'Gradient-based':>18}")
    print("-" * 48)
    for i in range(d1):
        lag = i - (d1 - 1)
        print(f"{lag:5d} | {weight_imp_g1[i]*100:7.2f}% | {grad_imp_g1[i]*100:8.2f}%")

    # ==================== DNN_g2 IMPORTANCE ====================
    print("\n" + "="*70)
    print("INPUT IMPORTANCE: DNN_g2 (Medium-frequency → Low factors)")
    print(f"d2 = {d2} lags (from lag -{d2-1} to lag 0)")
    print("="*70)

    # Weight-based
    w_g2 = get_first_linear_weight(DNN_g2)
    if w_g2 is not None:
        weight_imp_g2 = torch.abs(w_g2).mean(dim=0).cpu().detach().numpy()
        weight_imp_g2 /= (weight_imp_g2.sum() + 1e-12)
    else:
        weight_imp_g2 = np.zeros(d2)
        print("Warning: No Linear layer in DNN_g2 → weight importance unavailable")

    # Gradient-based
    grad_imp_g2 = np.zeros(d2)
    count = 0
    for p in range(D2):
        x = torch.FloatTensor(X_M_bar_reshape[p]).to(device)
        x.requires_grad_(True)
        out = DNN_g2(x)                  # (T, r2)
        out.abs().sum().backward()
        grad_imp_g2 += x.grad.abs().mean(dim=0).cpu().numpy()
        count += 1
        x.grad.zero_()
    grad_imp_g2 /= max(count, 1)
    grad_imp_g2 /= (grad_imp_g2.sum() + 1e-12)

    # Print table
    print(f"{'Lag':>5} | {'Weight-based':>16} | {'Gradient-based':>18}")
    print("-" * 48)
    for i in range(d2):
        lag = i - (d2 - 1)
        print(f"{lag:5d} | {weight_imp_g2[i]*100:7.2f}% | {grad_imp_g2[i]*100:8.2f}%")

################################ MIDAS ################################
def nealmon_weights_t(params, n_lags, device):
    # params: iterable-like or torch tensor [theta, decay]
    # n_lags: int (or convertible), device: torch.device
    params_t = torch.as_tensor(params, device=device, dtype=torch.float32)
    theta = params_t[0]
    decay = params_t[1]
    n = int(n_lags)
    if n < 1:
        raise ValueError("n_lags must be >= 1")
    lags = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    w = torch.exp(theta * lags + decay * lags * lags)
    w = w / (w.sum() + 1e-12)
    return w

def beta_weights_safe_t(params, n_lags, device, delta=0.01):
    # params: iterable-like or torch tensor [theta1, theta2]
    params_t = torch.as_tensor(params, device=device, dtype=torch.float32)
    theta1 = params_t[0]
    theta2 = params_t[1]
    K = int(n_lags)
    if K < 1:
        raise ValueError("n_lags must be >= 1")
    if K == 1:
        return torch.tensor([1.0], device=device, dtype=torch.float32)
    j = torch.arange(0, K, dtype=torch.float32, device=device)
    z = (j + delta) / (K - 1 + 2*delta)
    # compute beta-like shape safely
    w = (z ** (theta1 - 1.0)) * ((1.0 - z) ** (theta2 - 1.0))
    # avoid NaNs / divide-by-zero
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = w / (w.sum() + 1e-12)
    return w

def beta_restricted_weights_t(theta1, n_lags, device='cpu', delta=0.01, normalize=True):
    """
    Compute weights using φ(k, θ₁) = θ₁ * (1 - k)^(θ₁ - 1)
    
    Args:
        theta1 (float): Shape parameter θ₁
        n_lags (int): Number of lags (K)
        device (str): 'cpu' or 'cuda'
        delta (float): Smoothing offset to avoid k=0 or k=1
        normalize (bool): Whether to normalize weights to sum to 1

    Returns:
        torch.Tensor: Weight vector of shape (n_lags,)
    """
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1")
    if n_lags == 1:
        return torch.tensor([1.0], device=device, dtype=torch.float32)

    j = torch.arange(0, n_lags, dtype=torch.float32, device=device)
    k = (j + delta) / (n_lags - 1 + 2 * delta)
    w = theta1 * (1.0 - k) ** (theta1 - 1.0)
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize:
        w = w / (w.sum() + 1e-12)
    return w

def three_freq_midas_lagged_regression(Y, X_M, X_H, d1, d2, k, Weight_choice='Nealmon', device='cpu', Q=-1
                                       , method='Powell', inits=[2,2,2,2]):
    """
    Y: [P3, T] numpy
    X_M: [P2, T*d2] numpy
    X_H: [P1, T*d1*d2] numpy
    k: number of low-frequency lags to use
    """
    P1, Td1d2 = X_H.shape
    P2, Td2 = X_M.shape
    P3, T = Y.shape
    # print('Td1d2',Td1d2)
    # print('T * d1 * d2',T * d1 * d2)
    assert Td1d2 == T * d1 * d2
    assert Td2 == T * d2
    assert T > k, "Not enough time periods for lagged regression"

    # Convert to torch tensors on GPU
    Y_torch = torch.from_numpy(Y).float().to(device)
    X_H_torch = torch.from_numpy(X_H).float().to(device)
    X_M_torch = torch.from_numpy(X_M).float().to(device)

    # Reshape to [P1, T, d2, d1] and [P2, T, d2]
    X_H_reshaped = X_H_torch.reshape(P1, T, d2, d1)
    X_M_reshaped = X_M_torch.reshape(P2, T, d2)
    
    def objective(params):
        # params is numpy array from scipy.optimize
        params_torch = torch.tensor(params, device=device, dtype=torch.float32)
        params_high, params_med = params_torch[:2], params_torch[2:]
        
        if Weight_choice == 'Beta':
            w_high = beta_weights_safe_t(params_high, d1, device).flip(0)
            w_med = beta_weights_safe_t(params_med, d2, device).flip(0)
        elif Weight_choice == 'Nealmon':
            w_high = nealmon_weights_t(params_high, d1, device).flip(0)
            w_med = nealmon_weights_t(params_med, d2, device).flip(0)
        # Apply weights: high → medium [P1, T, d2]
        X_H_med = torch.tensordot(X_H_reshaped, w_high, dims=1)
        
        # Combine with medium [P2+P1, T, d2]
        X_M_combined = torch.cat([X_M_reshaped, X_H_med], dim=0)
        
        # Apply weights: medium → low [P2+P1, T]
        X_low_freq = torch.tensordot(X_M_combined, w_med, dims=1)
        
        # Combine with Y [P3+P2+P1, T]
        X_low_freq = torch.cat([Y_torch, X_low_freq], dim=0)
        
        # Build lagged regressors
        T_k = T - k
        X_final = torch.zeros((P3 + P2 + P1) * k, T_k, device=device)
        for i in range(T_k):
            X_final[:, i] = X_low_freq[:, i:i+k].flatten()
        
        Y_final = Y_torch[:, k:]  # [P3, T-k]
        
        # Convert to numpy for OLS
        X_final_np = X_final.cpu().numpy().T  # [T-k, (P3+P2+P1)*k]
        if Q == -1:
            Y_final_np = Y_final.cpu().numpy().T  # [T-k, P3]
        else:
            Y_final_np = Y_final.cpu().numpy().T[:,:Q]  # [T-k, P3]
        
        model = sm.OLS(Y_final_np, sm.add_constant(X_final_np)).fit()
        return np.sum((Y_final_np - model.fittedvalues)**2)

    if Weight_choice == 'Beta':
        # bounds = [(0.01, 3), (0.01, 3), (0.01, 3), (0.01, 3)]
        bounds = [(1, 3), (1, 3), (1, 3), (1, 3)]
        result = minimize(objective, inits, method=method, bounds=bounds)
    elif Weight_choice == 'Nealmon':
        if d1 > 10:
            bounds = [(-3, 3), (-0.03, 0.03), (-4, 4), (-3, 3)]
            if inits[1] > 0.05:
                inits[1] = 0.01
            elif inits[1] < -0.05:
                inits[1] = -0.01
        else:
            bounds = [(-5, 5), (-4, 4), (-5, 5), (-4, 4)]
        result = minimize(objective, inits, method=method, bounds=bounds)
    
    # Get final weights
    params_torch = torch.tensor(result.x, device=device, dtype=torch.float32)
    params_high, params_med = params_torch[:2], params_torch[2:]

    # Final computation for model (using GPU)
    if Weight_choice == 'Beta':
        w_high_t = beta_weights_safe_t(params_high, d1, device).flip(0)
        w_med_t = beta_weights_safe_t(params_med, d2, device).flip(0)
    elif Weight_choice == 'Nealmon':
        w_high_t = nealmon_weights_t(params_high, d1, device).flip(0)
        w_med_t = nealmon_weights_t(params_med, d2, device).flip(0)    
    # print('w_high', w_high_t)
    # print('w_med', w_med_t)
    
    # Apply weights: high → medium
    X_H_med = torch.tensordot(X_H_reshaped, w_high_t, dims=1)
    
    # Combine with medium
    X_M_combined = torch.cat([X_M_reshaped, X_H_med], dim=0)
    
    # Apply weights: medium → low
    X_low_freq = torch.tensordot(X_M_combined, w_med_t, dims=1)
    
    # Combine with Y
    X_low_freq = torch.cat([Y_torch, X_low_freq], dim=0)
    
    # Build lagged regressors
    T_k = T - k
    X_final = torch.zeros((P3 + P2 + P1) * k, T_k, device=device)
    for i in range(T_k):
        X_final[:, i] = X_low_freq[:, i:i+k].flatten()
    
    Y_final = Y_torch[:, k:]
    
    # Convert to numpy for OLS
    X_final_np = X_final.cpu().numpy().T
    if Q == -1:
        Y_final_np = Y_final.cpu().numpy().T  # [T-k, P3]
    else:
        Y_final_np = Y_final.cpu().numpy().T[:,:Q]  # [T-k, P3]
    

    model = sm.OLS(Y_final_np, sm.add_constant(X_final_np)).fit()

    w_high = w_high_t.cpu().numpy()
    w_med = w_med_t.cpu().numpy()
    return model, w_high, w_med

def midas_forecast_low_only(model, X_low, X_med, X_high, k, d1, d2, w_high, w_med, device='cpu'):
    """Forecast using GPU tensors."""
    P1 = X_high.shape[0]
    P2 = X_med.shape[0]
    
    # Convert to torch
    X_high_t = torch.from_numpy(X_high).to(device=device, dtype=torch.float32)
    X_med_t  = torch.from_numpy(X_med) .to(device=device, dtype=torch.float32)
    X_low_t  = torch.from_numpy(X_low) .to(device=device, dtype=torch.float32)
    
    w_high_t = torch.from_numpy(w_high).float().to(device)
    w_med_t = torch.from_numpy(w_med).float().to(device)
    
    X_H_reshaped = X_high_t.reshape(P1, -1, d2, d1)
    X_M_reshaped = X_med_t.reshape(P2, -1, d2)
    
    # Apply weights: high → medium
    X_H_med = torch.tensordot(X_H_reshaped, w_high_t, dims=1)
    
    # Combine with medium
    X_M_combined = torch.cat([X_M_reshaped, X_H_med], dim=0)
    
    # Apply weights: medium → low
    X_low_freq = torch.tensordot(X_M_combined, w_med_t, dims=1)
    
    # Combine with low frequency data
    X_low_freq = torch.cat([X_low_t, X_low_freq], dim=0)

    X_pred = X_low_freq.flatten()
    X_pred_np = torch.cat([torch.tensor([1.0], device=device), X_pred]).cpu().numpy().reshape(1, -1)
    forecast = model.predict(X_pred_np).reshape(-1)
    return forecast

def two_freq_midas_lagged_regression(Y, X_M, d, k, Weight_choice='Nealmon', device='cpu', Q1=5,
                                    method='Powell', inits=[0,0]):
    """
    Two-frequency MIDAS: High-frequency X_M -> Low-frequency Y

    Parameters:
    -----------
    Y : ndarray, shape (P_y, T)
        Low-frequency target variable(s)
    X_M : ndarray, shape (P_x, T*d)
        High-frequency predictor(s), d times per low-frequency period
    d : int
        Number of high-frequency periods per low-frequency period (e.g., 3 for monthly→quarterly)
    k : int
        Number of low-frequency lags in the regression
    Weight_choice : str
        'Nealmon' or 'Beta'
    device : str or torch.device
        'cpu' or 'cuda'

    Returns:
    --------
    model : statsmodels OLS result
    w_med : numpy array, final high-frequency weights
    """
    P_y, T = Y.shape
    P_x, Td = X_M.shape
    assert Td == T * d, f"X_M must have T*d={T*d} columns, got {Td}"
    assert T > k, "Not enough time periods for lagged regression"

    # Convert to torch
    Y_t = torch.from_numpy(Y).float().to(device)
    X_M_t = torch.from_numpy(X_M).float().to(device)

    # Reshape X_M to [P_x, T, d]
    X_M_reshaped = X_M_t.reshape(P_x, T, d)

    def objective(params):
        params_t = torch.tensor(params, device=device, dtype=torch.float32)
        
        if Weight_choice == 'Beta':
            w_med = beta_weights_safe_t(params_t, d, device).flip(0)
        elif Weight_choice == 'Nealmon':
            w_med = nealmon_weights_t(params_t, d, device).flip(0)

        # Aggregate high → low: [P_x, T]
        X_low = torch.tensordot(X_M_reshaped, w_med, dims=1)

        # Stack Y and aggregated X: [P_y + P_x, T]
        Z = torch.cat([Y_t, X_low], dim=0)

        # Build lagged regressors
        T_k = T - k
        P_total = P_y + P_x
        X_lagged = torch.zeros(P_total * k, T_k, device=device)
        for i in range(T_k):
            X_lagged[:, i] = Z[:, i:i+k].flatten()

        Y_lagged = Y_t[:, k:]  # [P_y, T-k]

        # Convert to numpy for OLS
        X_np = X_lagged.cpu().numpy().T  # [T-k, P_total*k]
        Y_np = Y_lagged.cpu().numpy().T[:,:Q1]  # [T-k, P_y]

        model = sm.OLS(Y_np, sm.add_constant(X_np)).fit()
        return np.sum((Y_np - model.fittedvalues) ** 2)

    # Optimization setup
    if Weight_choice == 'Beta':
        bounds = [(1, 3), (1, 3)]
    elif Weight_choice == 'Nealmon':
        bounds = [(-5, 5), (-2, 2)] if d <= 10 else [(-4, 4), (-0.1, 0.1)]

    result = minimize(objective, inits, method=method, bounds=bounds if method == 'L-BFGS-B' else None)

    # Final weights
    params_final = torch.tensor(result.x, device=device, dtype=torch.float32)
    if Weight_choice == 'Beta':
        w_med_t = beta_weights_safe_t(params_final, d, device).flip(0)
    else:
        w_med_t = nealmon_weights_t(params_final, d, device).flip(0)

    # Final model fit
    X_low = torch.tensordot(X_M_reshaped, w_med_t, dims=1)
    Z = torch.cat([Y_t, X_low], dim=0)
    T_k = T - k
    P_total = P_y + P_x
    X_lagged = torch.zeros(P_total * k, T_k, device=device)
    for i in range(T_k):
        X_lagged[:, i] = Z[:, i:i+k].flatten()
    Y_lagged = Y_t[:, k:]

    X_np = X_lagged.cpu().numpy().T
    Y_np = Y_lagged.cpu().numpy().T[:,:Q1]
    model = sm.OLS(Y_np, sm.add_constant(X_np)).fit()

    w_med = w_med_t.cpu().numpy()

    print(f"Optimized params: {np.round(result.x, 4)}")
    print(f"w_med (sum={w_med.sum():.4f}): {np.round(w_med, 4)}")

    return model, w_med

def midas_forecast_two_freq_one_step(
    model, Y_hist, X_M_block, d, k, w_med, device='cpu'
):
    """
    One-step-ahead forecast for the two-frequency MIDAS model.

    Parameters
    ----------
    model : statsmodels OLS result
        Fitted model from `two_freq_midas_lagged_regression`.
    Y_hist : ndarray, shape (P_y, k)
        The *last k* low-frequency observations of the target.
    X_M_block : ndarray, shape (P_x, d)
        The *current* high-frequency block that belongs to the period
        we want to forecast (the most recent d high-frequency observations).
    d : int
        Number of high-frequency periods per low-frequency period.
    k : int
        Number of low-frequency lags used in the regression.
    w_med : ndarray, shape (d,)
        Optimised MIDAS weights (high → low).
    device : str or torch.device
        'cpu' or 'cuda'.

    Returns
    -------
    forecast : ndarray, shape (P_y,)
        Forecast for the next low-frequency period.
    """
    P_y, T_hist = Y_hist.shape
    assert T_hist == k, f"Y_hist must contain exactly {k} low-frequency lags"

    # ------------------------------------------------------------------ #
    # 1. Convert everything to torch tensors
    # ------------------------------------------------------------------ #
    Y_t      = torch.from_numpy(Y_hist).float().to(device)          # (P_y, k)
    X_M_t    = torch.from_numpy(X_M_block).float().to(device)       # (P_x, d)
    w_med_t  = torch.from_numpy(np.tile(w_med, k)).float().to(device)           # (d,)

    # ------------------------------------------------------------------ #
    # 2. Aggregate the current high-frequency block → low-frequency
    # ------------------------------------------------------------------ #
    X_low_t = (X_M_t * w_med_t).sum(dim=1, keepdim=True)            # (P_x, 1)

    # ------------------------------------------------------------------ #
    # 3. Build the regressor vector:  [1 | Y_{t-1} … Y_{t-k} |
    #                                    X_{t-1} … X_{t-k} | X_t ]
    # ------------------------------------------------------------------ #
    Z_lags = []
    for lag in range(k):
        # lag = 0  → most recent *known* low-frequency period (t-1)
        # lag = k-1→ oldest lag (t-k)
        Y_lag = Y_t[:, lag : lag + 1]                               # (P_y, 1)

        if lag == 0:                                   # most recent known X
            X_low_lag = X_low_t
        else:
            X_low_lag = torch.zeros_like(X_low_t)

        Z_lags.append(torch.cat([Y_lag, X_low_lag], dim=0))        # (P_y+P_x, 1)

    # Concatenate all k lags → (P_y+P_x, k) then flatten
    X_reg = torch.cat(Z_lags, dim=1).flatten()                     # ( (P_y+P_x)*k ,)

    # Add constant term
    X_reg = torch.cat([torch.tensor([1.0], device=device), X_reg])  # (1 + (P_y+P_x)*k ,)

    # ------------------------------------------------------------------ #
    # 4. Predict with the OLS coefficients
    # ------------------------------------------------------------------ #
    coeffs = np.concatenate([model.params[:1], model.params[1:]])   # (1 + total_vars,)
    forecast = X_reg.cpu().numpy() @ coeffs                                 # (P_y,)

    return forecast


# =============================
# MIDAS+Unified Final Regressor (DNN, RNN, LSTM, GRU, Transformer)
# =============================
class FinalRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, model_type='dnn', 
                 hidden_dim=256, seq_len=1, num_layers=2, dropout=0.3, nhead=8):
        super().__init__()
        self.model_type = model_type.lower()
        self.seq_len = seq_len
        self.features_per_step = input_dim // seq_len  # = D (full features per low-freq period)

        # ------------------- DNN (updated style — like DNN3_G3) -------------------
        if self.model_type == 'dnn':
            W = hidden_dim
            L = num_layers
            
            if L < 1:
                raise ValueError("dnn_layers must be at least 1")
            
            # Store parameters
            self.L = L
            self.W = W
            
            # First layer
            self.fc1 = nn.Linear(input_dim, W)
            
            # Middle hidden layers (fc2 → fc{L})
            self.mid_layers = nn.ModuleList([
                nn.Linear(W, W) for _ in range(L-1)
            ])
            
            # Output layer
            self.fc_out = nn.Linear(W, output_dim)
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout)
        
        # ------------------- RNN / LSTM / GRU -------------------
        elif self.model_type in ['rnn', 'lstm', 'gru']:
            rnn_class = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[self.model_type]
            self.rnn = rnn_class(
                input_size=self.features_per_step,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.fc = nn.Linear(hidden_dim, output_dim)

        # ------------------- Transformer -------------------
        elif self.model_type == 'transformer':
            D = self.features_per_step

            # Make embed_dim safely divisible by nhead
            embed_dim = (D // nhead) * nhead
            if embed_dim < nhead * 4:
                embed_dim = max(nhead * 4, 64)

            self.input_proj = nn.Linear(D, embed_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(embed_dim, output_dim)

        else:
            raise ValueError("model_type must be 'dnn', 'rnn', 'lstm', 'gru', or 'transformer'")

    def forward(self, x):
        if self.model_type == 'dnn':
            # DNN forward pass (flexible depth)
            x = self.relu(self.fc1(x))
            
            for i in range(self.L - 1):
                x = self.dropout(x)
                x = self.relu(self.mid_layers[i](x))
            
            # No dropout before final linear layer
            x = self.fc_out(x)
            return x

        # Reshape once — shared for sequential models
        batch_size = x.size(0)
        x_seq = x.view(batch_size, self.seq_len, self.features_per_step)

        if self.model_type in ['rnn', 'lstm', 'gru']:
            out, _ = self.rnn(x_seq)
            out = out[:, -1, :]           # last time step
            return self.fc(out)

        elif self.model_type == 'transformer':
            x = self.input_proj(x_seq)    # (B, seq_len, embed_dim)
            out = self.transformer(x)
            out = out[:, -1, :]           # last position
            return self.fc(out)

# =============================
# MIDAS + Neural Training Function
# =============================
def three_freq_midas_lagged_neural(
    Y, X_M, X_H, d1, d2, k,
    Weight_choice='Nealmon',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    Q=-1,
    final_model='dnn',           # 'dnn', 'rnn', 'lstm', 'gru', 'transformer'
    hidden_dim=512,
    num_layers=2,
    dropout=0.1,
    nhead=8,
    epochs=200,
    batch_size=64,
    lr=1e-3,
    patience=30,
    verbose=True,
    eval_horizon = 1,
    method = 'L-BFGS-B',    # 'L-BFGS-B' or 'Powell'
    inits = [0,0,0,0]  
):
    # target horizon (1-based)
    target_horizon = int(eval_horizon) + 1
    P1, Td1d2 = X_H.shape
    P2, Td2 = X_M.shape
    P3, T = Y.shape

    assert Td1d2 == T * d1 * d2 and Td2 == T * d2 and T > k

    Y_torch = torch.from_numpy(Y).float().to(device)
    X_H_torch = torch.from_numpy(X_H).float().to(device)
    X_M_torch = torch.from_numpy(X_M).float().to(device)

    X_H_reshaped = X_H_torch.reshape(P1, T, d2, d1)
    X_M_reshaped = X_M_torch.reshape(P2, T, d2)

    # ------------------- Optimize MIDAS weights -------------------
    def objective(params):
        params_t = torch.tensor(params, device=device, dtype=torch.float32)
        params_high, params_med = params_t[:2], params_t[2:]

        if Weight_choice == 'Beta':
            w_high = beta_weights_safe_t(params_high, d1, device).flip(0)
            w_med = beta_weights_safe_t(params_med, d2, device).flip(0)
        else:
            w_high = nealmon_weights_t(params_high, d1, device).flip(0)
            w_med = nealmon_weights_t(params_med, d2, device).flip(0)

        X_H_med = torch.tensordot(X_H_reshaped, w_high, dims=1)
        X_M_combined = torch.cat([X_M_reshaped, X_H_med], dim=0)
        X_low_freq = torch.tensordot(X_M_combined, w_med, dims=1)
        X_low_freq = torch.cat([Y_torch, X_low_freq], dim=0)

        T_k_h = T - k - target_horizon + 1
        X_lagged = torch.zeros((P3 + P2 + P1) * k, T_k_h, device=device)
        for i in range(T_k_h):
            X_lagged[:, i] = X_low_freq[:, i:i+k].flatten()

        Y_target = Y_torch[:, k+target_horizon-1:]

        X_np = X_lagged.cpu().numpy().T
        Y_np = Y_target.cpu().numpy().T[:, :Q] if Q != -1 else Y_target.cpu().numpy().T

        model = sm.OLS(Y_np, sm.add_constant(X_np)).fit()
        return np.sum((Y_np - model.fittedvalues)**2)

    if Weight_choice == 'Beta':
        bounds = [(0.01, 3)] * 4
        result = minimize(objective, inits, method=method, bounds=bounds)
    else:
        bounds = [(-5, 5), (-3, 3), (-5, 5), (-4, 4)] if d1 <= 10 else [(-2, 2), (-0.04, 0.04), (-4, 4), (-3, 3)]
        result = minimize(objective, inits, method=method, bounds=bounds)

    # ------------------- Apply optimal weights -------------------
    params_t = torch.tensor(result.x, device=device, dtype=torch.float32)
    params_high, params_med = params_t[:2], params_t[2:]

    if Weight_choice == 'Beta':
        w_high_t = beta_weights_safe_t(params_high, d1, device).flip(0)
        w_med_t = beta_weights_safe_t(params_med, d2, device).flip(0)
    else:
        w_high_t = nealmon_weights_t(params_high, d1, device).flip(0)
        w_med_t = nealmon_weights_t(params_med, d2, device).flip(0)

    X_H_med = torch.tensordot(X_H_reshaped, w_high_t, dims=1)
    X_M_combined = torch.cat([X_M_reshaped, X_H_med], dim=0)
    X_low_freq = torch.tensordot(X_M_combined, w_med_t, dims=1)
    X_low_freq = torch.cat([Y_torch, X_low_freq], dim=0)  # (D, T)

    # ------------------- Build lagged dataset -------------------
    T_k_h = T - k - target_horizon + 1
    D = P3 + P2 + P1
    X_lagged = torch.zeros(D * k, T_k_h, device=device)
    for i in range(T_k_h):
        X_lagged[:, i] = X_low_freq[:, i:i+k].flatten()

    X_data = X_lagged.T  # (T_k, D*k)
    Y_data = Y_torch[:, k+target_horizon-1:].T
    if Q != -1:
        Y_data = Y_data[:, :Q]

    # ------------------- Train neural regressor -------------------
    model = FinalRegressor(
        input_dim=D * k,
        output_dim=Y_data.shape[1],
        model_type=final_model,
        hidden_dim=hidden_dim,
        seq_len=k,
        num_layers=num_layers,
        dropout=dropout,
        nhead=nhead
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    dataset = TensorDataset(X_data, Y_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    best_state = None
    counter = 0

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(dataset)

        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            best_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}, best loss: {best_loss:.6f}")
            break

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        fitted = model(X_data).cpu().numpy()

    w_high = w_high_t.cpu().numpy()
    w_med = w_med_t.cpu().numpy()

    if verbose:
        print(f"Final {final_model.upper()} MSE: {best_loss:.6f}")

    return model, fitted, w_high, w_med


# =============================
# Forecasting Function (works with all models)
# =============================
def midas_forecast_neural(
    model,
    X_low_hist,      # (P3, k) — past k values of Y
    X_med,           # (P2, k*d2)
    X_high,          # (P1, k*d1*d2)
    k, d1, d2,
    w_high, w_med,
    device='cpu'
):
    model.eval()
    with torch.no_grad():
        P1 = X_high.shape[0]
        P2 = X_med.shape[0]
        P3 = X_low_hist.shape[0]
        D = P3 + P2 + P1

        X_high_t = torch.from_numpy(X_high).float().to(device)
        X_med_t = torch.from_numpy(X_med).float().to(device)
        X_low_hist_t = torch.from_numpy(X_low_hist).float().to(device)

        w_high_t = torch.from_numpy(w_high).float().to(device)
        w_med_t = torch.from_numpy(w_med).float().to(device)

        X_H_reshaped = X_high_t.reshape(P1, k, d2, d1)
        X_M_reshaped = X_med_t.reshape(P2, k, d2)

        X_H_med = torch.tensordot(X_H_reshaped, w_high_t, dims=([3], [0]))
        X_M_combined = torch.cat([X_M_reshaped, X_H_med], dim=0)
        X_low_proj = torch.tensordot(X_M_combined, w_med_t, dims=([2], [0])) #(P1+P2, k)

        # Build k-lag low-frequency vector
        current_low = torch.cat([
            X_low_hist_t,                    # (P3, k-1)
            X_low_proj       # (P1+P2, 1)
        ], dim=0)                                      # (D, k)

        X_input = current_low.flatten().unsqueeze(0).to(device)  # (1, D*k)

        forecast = model(X_input).cpu().numpy().flatten()
    return forecast
