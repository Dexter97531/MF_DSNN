# MF_DSNN
**Mixed-Frequency Time Series Forecasting via Depth-Separable Neural Networks**

This repository implements **DSNN** (Depth-Separable Neural Networks) for mixed-frequency time series forecasting, along with several baseline algorithms for comparison. It also includes real-world datasets preprocessed from **FRED (Federal Reserve Economic Data)**.

---

## 📂 Repository Structure

### **Method/**
- `Baseline.py`  
  Contains implementations of:
  - **DSNN**  
  - **MIDAS** (Mixed Data Sampling regression)

---

### **baseline/**
Baseline models for comparison:
- **BMF** (Chakraborty et al., 2023)  
- **Seq2one**  (Lin & Michailidis, 2024)
- **Transformer** (Lin & Michailidis, 2024)

---

### **real_data/real_data_DMQ_7301_2412_56/**
- **data_clean/**  
  Preprocessed datasets ready for modeling.
- **model_std/**  
  Standard model implementations:
  - DSNN  
  - Stacking + VARX  
  - Stacking + Neural Networks (DNN, RNN, LSTM)  
  - MIDAS + VARX  
  - MIDAS + Neural Networks  

---

### **utils.py**
Utility functions for:
- Data reshaping  
- Metrics evaluation  
- Standardization  
- Plotting  

---

