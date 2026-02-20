# MF_DSNN
Mixed-Frequency Time Series Forecasting via Depth-Separable Neural Networks

**./Method/Baseline.py**
contains DSNN and MIDAS algorithms

**./baseline**
contains BMF(Chakraborty et al, 2023), Seq2one and Transformer (Lin and Michailidis, 2024)

**./real_data/real_data_DMQ_7301_2412_56/
  ./data_clean**
    after preprocessed data
  **./model_std**
    codes for DSNN, Stacking+VARX, Stacking+Neural Networks(DNN,RNN,LSTM), MIDAS+VARX, MIDAS+Neural Networks.

**./utils.py**
additional functions including reshaping, metrics, standardizing and ploting
