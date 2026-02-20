"""
Trainer class for running DNN models on real data
"""

import sys
import os
import shutil
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from models import MTMFSeq2One, MTMFSeq2OnePred, Transformer
from helpers import ClsConstructor
from .utils_train import *

class LossPrintCallback(tf.keras.callbacks.Callback):
    def __init__(self,every_n_epochs=100):
        super(LossPrintCallback,self).__init__()
        self.every_n_epochs = every_n_epochs
    def on_epoch_end(self,epoch,logs=None):
        if logs.get('output_1_loss') is not None and logs.get('output_2_loss') is not None:
            if (epoch+1)%self.every_n_epochs == 0:
                print(f"  >> epoch = {epoch+1}; loss = {logs.get('loss'):.4f}; output_1_loss = {logs.get('output_1_loss'):.4f}, output_2_loss = {logs.get('output_2_loss'):.4f}.")
        else:
            if (epoch+1)%self.every_n_epochs == 0:
                print(f"  >> epoch = {epoch+1}; loss = {logs.get('loss'):.4f}.")

class nnTrainer():

    def __init__(self,args,criterion=tf.keras.losses.MeanSquaredError(),seed=411):
        
        self.args = args
        self.criterion = criterion
        self.cls_constructor = ClsConstructor(self.args)
        self.seed = seed
    
    def set_seed(self, repickle_args=True):
        
        #tf.keras.utils.set_random_seed(self.seed)
        tf.random.set_seed(self.seed)
        
        setattr(self.args, 'seed', self.seed)
        with open(f"{self.args.output_folder}/args.pickle","wb") as handle:
            pickle.dump(self.args, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    def source_data(self):
        
        # x = pd.read_excel(f"{self.args.data_folder}/{self.args.ds_name}.xlsx",index_col='timestamp',sheet_name='x')
        # y = pd.read_excel(f"{self.args.data_folder}/{self.args.ds_name}.xlsx",index_col='timestamp',sheet_name='y')
        # x.index, y.index = pd.to_datetime(x.index), pd.to_datetime(y.index)
        # xdata, ydata = x.values, y.values
        
        # self.df_info = {'x_index': list(x.index),
        #                 'y_index': list(y.index),
        #                 'x_columns': list(x.columns),
        #                 'y_columns': list(y.columns),
        #                 'x_total_obs': x.shape[0],
        #                 'y_total_obs': y.shape[0]
        #                }
        # self.raw_data = (xdata, ydata)
        data_full = pd.read_csv("D:/Code/MF/data/data3_197301_202503_56_9_std_reorder_v2.csv", header=None)

        # Generate timestamps
        # start_date = pd.to_datetime("1960-01-01")
        start_date = pd.to_datetime("1973-01-01")
        n_time_points = data_full.shape[1]
        monthly_timestamps = pd.date_range(start=start_date, periods=3*n_time_points, freq='M')
        quarterly_timestamps = pd.date_range(start=start_date, periods=n_time_points, freq='Q')

        # Split data into high-frequency (x) and low-frequency (y)
        P = 56  # Number of high-frequency variables (from R code)
        Q = 9   # Number of low-frequency variables (from R code)
        xdata = data_full.iloc[:(3*P), :].values  # High-frequency variables (monthly)
        ydata = data_full.iloc[(3*P):(3*P+Q), :].values  # Low-frequency variables (quarterly)
        # Step 1: Reshape 72 monthly series to 24 variables
        monthly_data = xdata  # Shape: (72, 239)
        n_time_points = monthly_data.shape[1] * 3  # 239 quarters * 3 months = 717
        monthly_reshaped = np.zeros((P, n_time_points))
        for i in range(P):
            for j in range(monthly_data.shape[1]):  # For each quarter
                monthly_reshaped[i, j*3:(j+1)*3] = monthly_data[3*i:3*(i+1), j]
        xdata = monthly_reshaped

        # Create DataFrames for metadata
        # print('monthly_timestamps',monthly_timestamps.shape)

        x = pd.DataFrame(xdata.T, index=monthly_timestamps, columns=[f"hf_{i}" for i in range(P)])
        y = pd.DataFrame(ydata.T, index=quarterly_timestamps, columns=[f"lf_{i}" for i in range(Q)])

        # Store metadata
        self.df_info = {
            'x_index': list(x.index),
            'y_index': list(y.index),
            'x_columns': list(x.columns),
            'y_columns': list(y.columns),
            'x_total_obs': x.shape[0],
            'y_total_obs': y.shape[0]
        }
        self.raw_data = (xdata.T, ydata.T)  # Transpose to match expected shape (time, features)
        
    def generate_train_val_datasets(self, x_train_end, y_train_end, n_val = None):
        """
        helper function for generating train/val dataset; the reason for not adding them as attributes is out of
        consideration for dynamic run
        Argv:
        - x_train_end, y_train_end: the ending index, resp for x and y
        """
        args = self.args
        dp = self.cls_constructor.create_data_processor()
        
        xdata, ydata = self.raw_data
        
        ## get training data
        x_train, y_train = xdata[:x_train_end,:], ydata[:y_train_end,:]
        print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
        train_inputs, train_targets = dp.mf_sample_generator(x_train,
                                                             y_train,
                                                             update_scaler = True if args.scale_data else False,
                                                             apply_scaler = True if args.scale_data else False)

        print(f'[train samples generated] inputs dims: {[temp.shape for temp in train_inputs]}; target dims: {[temp.shape for temp in train_targets]} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        if n_val is not None:
            if isinstance(n_val, float):
                n_val = round(train_inputs[0].shape[0] * n_val)
            
            train_inputs, val_inputs = [x[:-n_val] for x in train_inputs], [x[-n_val:] for x in train_inputs]
            train_targets, val_targets = [x[:-n_val] for x in train_targets], [x[-n_val:] for x in train_targets]
            print(f'[{n_val} val samples reserved] inputs dims: {[temp.shape for temp in val_inputs]}; target dims: {[temp.shape for temp in val_targets]} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        train_data = (train_inputs, train_targets)
        val_data = (val_inputs, val_targets) if n_val is not None else None
        
        return dp, train_data, val_data
            
    def config_and_train_model(self, train_data, val_data = None):
        
        args = self.args
        # model = self.cls_constructor.create_model()
        if args.model_type == 'MTMFSeq2One':
            model = MTMFSeq2One(
                Lx= 3,
                Ty= 1,
                # Lx= 12,
                # Ty= 4,
                dim_x = 56,
                dim_y = 9,
                n_a=128,  # Adjust as per your config
                n_s=256,  # Adjust as per your config
                n_align=16,  # Adjust as per your config
                fc_x=256,  # Adjust as per your config
                fc_y=128,  # Adjust as per your config
                dropout_rate=0.4,
                freq_ratio=3,
                bidirectional_encoder=False,
                l1reg=0.0001,
                l2reg=0.0001,
                )
        elif args.model_type == 'transformer':
            model = Transformer(
                dim_x=56,  # Number of high-frequency variables
                dim_y=9,   # Number of low-frequency variables
                Tx=3,      # Should be same as freq_ratio
                # Lx=12,     # Length of input high-frequency sequence
                # Ty=4,      # Length of output low-frequency sequence
                Lx=3,     # Length of input high-frequency sequence
                Ty=1,      # Length of output low-frequency sequence
                key_dim_enc=16,    # Encoder key dimension
                fc_dim_enc=56,     # Encoder fully-connected layer dimension
                key_dim_xdec=16,   # x-decoder key dimension
                fc_dim_xdec=32,    # x-decoder fully-connected layer dimension
                ffn_dim_x=128,     # x-decoder final feed-forward network dimension
                key_dim_ydec=16,   # y-decoder key dimension
                fc_dim_ydec=32,    # y-decoder fully-connected layer dimension
                ffn_dim_y=128,     # y-decoder final feed-forward network dimension
                num_layers=1,      # Number of transformer layers
                num_heads=4,       # Number of attention heads
                freq_ratio=3,      # Frequency ratio (monthly to quarterly)
                dropout_rate=0.4,  # Dropout rate
                layernorm_eps=1e-6,  # Layer normalization epsilon
                bidirectional_encoder=True  # Use bidirectional encoder
        )

        
        # Build and inspect model
        # model.build(input_shape=[(None, 216, 56), (None, 24, 9)])
        # print("Model output names:", model.output_names)

        train_inputs, train_targets = train_data
        
        # with open(f'{args.output_folder}/model_summary.txt', 'w') as f:
            # model.build_graph().summary(print_fn=lambda x: f.write(x + '\n'))
        with open('model_summary.txt', 'w', encoding='utf-8') as f:
            model.build_graph().summary(print_fn=lambda x: f.write(x + '\n'))       
        ## set up callbacks
        callbacks = []
        if len(args.reduce_LR_monitor):
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=args.reduce_LR_monitor,
                                                             factor = args.reduce_LR_factor,
                                                             patience = args.reduce_LR_patience,
                                                             min_lr = 0.000001)
            callbacks.append(reduce_lr)
        if args.ES_patience is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(patience=args.ES_patience, monitor='val_loss',min_delta=0, mode='min')
            callbacks.append(early_stopping)
        if args.verbose > 0:
            loss_printer = LossPrintCallback(args.verbose)
            callbacks.append(loss_printer)
        
        ## compile
        print(f'[{args.model_type} model training starts] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        model.compile(loss = self.criterion,
                      optimizer = Adam(learning_rate=args.learning_rate),
                      metrics = [tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.RootMeanSquaredError()]
                    #   metrics={
                        # 'x_output': tf.keras.metrics.RootMeanSquaredError(name='rmse_x'),
                        # 'y_output': tf.keras.metrics.RootMeanSquaredError(name='rmse_y')
                    # }
                    )
        ## train
        history = model.fit(train_inputs,
                            train_targets,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            shuffle=args.shuffle,
                            validation_data=val_data,
                            callbacks=callbacks,
                            verbose=0)
                            
        print(f'[{args.model_type} model training ends] epoch = {len(history.history["loss"])} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        # plot_loss_over_epoch(history, args, save_as_file = f'{args.output_folder}/loss_over_epoch.png')
        return model
            
    def eval_train(self, model, dp, train_data):
        
        args = self.args
        train_inputs, train_targets = train_data
        
        x_is_pred, y_is_pred = model.predict(train_inputs)
        if args.scale_data:
            x_is_pred = dp.apply_scaler('scaler_x', x_is_pred, inverse=True)
            y_is_pred = dp.apply_scaler('scaler_y', y_is_pred, inverse=True)
            x_is_truth = dp.apply_scaler('scaler_x', train_targets[0], inverse=True)
            y_is_truth = dp.apply_scaler('scaler_y', train_targets[1], inverse=True)
        else:
            x_is_truth, y_is_truth = train_targets[0], train_targets[1]
            
        plot_fitted_val(args, (x_is_pred, y_is_pred), (x_is_truth, y_is_truth), x_col = self.df_info['x_columns'].index(args.X_COLNAME), y_col = self.df_info['y_columns'].index(args.Y_COLNAME), time_steps = None, save_as_file = f'{args.output_folder}/train_fit_static.png')
    
    def config_predictor(self, model, dp):
        
        args = self.args
        predictor = self.cls_constructor.create_predictor(model, dp, apply_inv_scaler = args.scale_data)
        
        return predictor
    
    def run_forecast_one_set(self, predictor, dp, y_start_id, x_start_id):
        
        """ helper function for running one set of forecast: F, N1, N2, N3 """
        args = self.args
        xdata, ydata = self.raw_data
        
        assert x_start_id == args.freq_ratio * (y_start_id - 1)
        predictions_by_vintage = {}
        
        for x_step in range(args.freq_ratio+1):
            experiment_tag = 'F' if x_step == 0 else f'N{x_step}'
            inputs, targets = dp.create_one_forecast_sample(xdata,
                                                            ydata,
                                                            x_start_id,
                                                            y_start_id,
                                                            x_step = x_step,
                                                            horizon = args.horizon ,
                                                            apply_scaler = True if args.scale_data else False,
                                                            verbose= False)
                                                            
            x_pred, y_pred = predictor(inputs, x_step = x_step, horizon = args.horizon)
            predictions_by_vintage[experiment_tag] = {'x_pred': x_pred, 'y_pred': y_pred}
        return predictions_by_vintage
    
    def add_prediction_to_collector(self, predictions_by_vintage, T_datestamp, x_PRED_collector = [], y_PRED_collector = []):
        
        args = self.args
        
        ## initialization for recording the forecast
        x_numeric_col_keys = [f'step_{i+1}' for i in range(args.freq_ratio * args.horizon)]
        y_numeric_col_keys = [f'step_{i+1}' for i in range(args.horizon)]
        
        ## extract prediction
        for vintage, predictions in predictions_by_vintage.items():
            x_pred, y_pred = predictions['x_pred'], predictions['y_pred']
            for col_id, variable_name in enumerate(self.df_info['x_columns']):
                temp = {'prev_QE': T_datestamp, 'tag': vintage, 'variable_name': variable_name}
                temp.update(dict(zip(x_numeric_col_keys, list(x_pred[:,col_id]))))
                x_PRED_collector.append(temp)
            for col_id, variable_name in enumerate(self.df_info['y_columns']):
                temp = {'prev_QE': T_datestamp, 'tag': vintage, 'variable_name': variable_name}
                temp.update(dict(zip(y_numeric_col_keys, list(y_pred[:,col_id]))))
                y_PRED_collector.append(temp)
        
    def run_forecast(self):
    
        """ main function """
    
        args = self.args
        ## initialization for recording the forecast
        x_PRED_collector, y_PRED_collector = [], []
        y_true_collector, y_pred_collector = [], []
        
        if args.mode == 'static':
            
            x_train_end = self.df_info['x_index'].index(args.first_prediction_date) - args.freq_ratio + 1 ## +1 to ensure the index is inclusive
            y_train_end = self.df_info['y_index'].index(args.first_prediction_date) - 2 + 1 ## +1 to ensure the index is inclusive
            
            ## set up and train model
            dp, train_data, val_data = self.generate_train_val_datasets(x_train_end, y_train_end, n_val = args.n_val)
            # print(f"train_data[1].shape (y): {train_data[1].shape}")
            model = self.config_and_train_model(train_data, val_data=val_data)
            # self.eval_train(model, dp, train_data)
            predictor = self.config_predictor(model, dp)
            
            ## run rolling forecast based on the trained model
            y_start_id = self.df_info['y_index'].index(args.first_prediction_date) - (args.Ty - 1)
            x_start_id = args.freq_ratio * (y_start_id - 1)
            # test_size = self.df_info['y_total_obs'] - self.df_info['y_index'].index(args.first_prediction_date)
            test_size = 20
            print('====test_size In dnn_trainer.py:', test_size)
            print(f'[forecast starts] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            for experiment_id in range(test_size):
                
                T_datestamp = self.df_info['x_index'][x_start_id + args.Lx - 1]
                x_range = [self.df_info['x_index'][x_start_id], self.df_info['x_index'][x_start_id + args.Lx - 1]]
                y_range = [self.df_info['y_index'][y_start_id], self.df_info['y_index'][y_start_id + args.Ty - 2]] ## -2 since Ty = Ly+1
                
                print(f" >> id = {experiment_id+1}/{test_size}: prev timestamp = {T_datestamp}; x_input_range = {x_range}, y_input_range = {y_range}")
                
                predictions_by_vintage = self.run_forecast_one_set(predictor, dp, y_start_id, x_start_id)
                self.add_prediction_to_collector(predictions_by_vintage, T_datestamp, x_PRED_collector, y_PRED_collector)
            
                # Collect true and predicted y values
                inputs, targets = dp.create_one_forecast_sample(
                    self.raw_data[0], self.raw_data[1], x_start_id, y_start_id,
                    x_step=0, horizon=args.horizon, apply_scaler=args.scale_data, verbose=False
                )
                x_pred, y_pred = predictions_by_vintage['F']['x_pred'], predictions_by_vintage['F']['y_pred']
                y_true = targets[1]  # Assuming targets[1] is y_target

                if args.scale_data:
                    y_true = dp.apply_scaler('scaler_y', y_true, inverse=True)
                print('y_true.shape:', y_true.shape)
                print('y_pred.shape:', y_pred.shape)
                print('y_true:', y_true)
                print('y_pred:', y_pred)
                y_true_collector.append(y_true)
                y_pred_collector.append(y_pred)

                # update x_start_id and y_start_id
                x_start_id += args.freq_ratio
                y_start_id += 1
                
            print(f'[forecast ends] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            
        elif args.mode == 'dynamic':
        
            offset = self.df_info['y_index'].index(args.first_prediction_date)
            # test_size = self.df_info['y_total_obs'] - offset
            test_size = 20
            
            for experiment_id in range(test_size):
                
                pred_date = self.df_info['y_index'][offset + experiment_id]
                
                print(f' >> id = {experiment_id+1}/{test_size}: next QE = {pred_date.strftime("%Y-%m-%d")}')
                ## set up and train model
                x_train_end = self.df_info['x_index'].index(pred_date) - args.freq_ratio + 1 ## +1 to ensure the index is inclusive
                y_train_end = self.df_info['y_index'].index(pred_date) - 2 + 1 ## +1 to ensure the index is inclusive
                dp, train_data, val_data = self.generate_train_val_datasets(x_train_end, y_train_end, n_val = args.n_val)
                model = self.config_and_train_model(train_data, val_data=val_data)
                # self.eval_train(model, dp, train_data)
                predictor = self.config_predictor(model, dp)
                
                ## run forecast
                y_start_id = self.df_info['y_index'].index(pred_date) - (args.Ty - 1)
                x_start_id = args.freq_ratio * (y_start_id - 1)
                T_datestamp = self.df_info['x_index'][x_start_id + args.Lx - 1]
                
                predictions_by_vintage = self.run_forecast_one_set(predictor, dp, y_start_id, x_start_id)
                self.add_prediction_to_collector(predictions_by_vintage, T_datestamp, x_PRED_collector, y_PRED_collector)
                
                # Collect true and predicted y values
                inputs, targets = dp.create_one_forecast_sample(
                    self.raw_data[0], self.raw_data[1], x_start_id, y_start_id,
                    x_step=0, horizon=args.horizon, apply_scaler=args.scale_data, verbose=False
                )
                x_pred, y_pred = predictions_by_vintage['F']['x_pred'], predictions_by_vintage['F']['y_pred']
                y_true = targets[1]
                if args.scale_data:
                    y_true = dp.apply_scaler('scaler_y', y_true, inverse=True)
                y_true_collector.append(y_true)
                y_pred_collector.append(y_pred)

                del predictor
                del model
                K.clear_session()

    # Convert collected y values to arrays for error computation
        y_true_mat = np.stack(y_true_collector, axis=0)  # Shape: (test_size, horizon, dim_y)
        y_pred_mat = np.stack(y_pred_collector, axis=0)  # Shape: (test_size, horizon, dim_y)
        
        # Compute RMSE and MAE
        def root_mean_squared_error(true, pred):
            squared_error = np.square(true - pred)
            mse_t = np.mean(squared_error, axis=0)  # Mean squared error for each quarter
            rmse_loss = np.sqrt(np.mean(mse_t))     # Mean RMSE across all quarters
            return mse_t, rmse_loss

        def mean_absolute_error(true, pred):
            mae_t = np.mean(np.abs(true - pred), axis=0)  # Mean absolute error for each quarter
            mae = np.mean(mae_t)                          # Mean MAE across all quarters
            return mae_t, mae

        y_true_mat = y_true_mat[:,:5]
        y_pred_mat = y_pred_mat[:,:5]
        err_MAE_t_nn, err_MAE_nn = mean_absolute_error(y_true_mat, y_pred_mat)
        err_t_nn, err_nn = root_mean_squared_error(y_true_mat, y_pred_mat)
    # Prepare output directory based on mode
        base_output_path = r"D:\Code\MF\baseline\2024_IJF\encoder-dual-decoder\output_FRED_COVID_56_lag1"
        print('==========================',f"{args.model_type}_{args.mode}")
        output_dir = os.path.join(base_output_path, f"{args.model_type}_{args.mode}")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "nn_results.txt")
        
        # Print and save results
        print("\n=== NN Option Results ===")
        print("True values:", y_true_mat.T)
        print("\nNN MAE:", err_MAE_t_nn)
        print("NN RMSE:", err_t_nn)
        print("NN Mean MAE:", err_MAE_nn)
        print("NN Mean RMSE:", err_nn)
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("=== NN Option Results ===\n")
            f.write(f"Model Mode: {args.model_type}_{args.mode}\n")
            f.write(f'[forecast ends] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            f.write(f"epochs: {args.epochs}\n")
            f.write(f"True values: {y_true_mat.T}\n") #(9.20)
            f.write(f"Pred values: {y_pred_mat.T}\n") #(9.20)
            f.write(f"\nNN MAE: {err_MAE_t_nn}\n")
            f.write(f"NN RMSE: {err_t_nn}\n")
            f.write(f"NN Mean MAE: {err_MAE_nn}\n")
            f.write(f"NN Mean RMSE: {err_nn}\n")

        x_PRED_df, y_PRED_df = pd.DataFrame(x_PRED_collector), pd.DataFrame(y_PRED_collector)
        
        with pd.ExcelWriter(args.output_filename) as writer:
            x_PRED_df.to_excel(writer,sheet_name=f'x_prediction',index=False)
            y_PRED_df.to_excel(writer,sheet_name=f'y_prediction',index=False)
