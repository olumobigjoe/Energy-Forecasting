# Kajola Village Hybrid Microgrid: SoC and Energy Forecasting for 2026/2027

**Hybrid Differencing + BiLSTM + ANN Model Using 2024/2025 Data**
A Case Study of Kajola Village, Sagamu, Ogun State

## 📌 Project Overview
This project forecasts State-of-Charge (SoC) and available energy for the planning horizon 2026/2027 using historical dataset generated for 2024/2025. The hybrid model achieves **0.39% RMSE and 0.26% MAE**, superior to conventional LSTM models.

Primary Aim: To predict energy storage availability for 2026/2027 when base load increases from 68kW to 72kW.

## 🧠 Model Architecture
- **Input:** 72-hour lookback window (Load, Solar Irradiance, Temp, Humidity, Hour, Day, Month)
- **Layer 1:** First-order Differencing (handles non-stationarity)
- **Layer 2:** BiLSTM 128 units + Dropout 0.2
- **Layer 3:** BiLSTM 64 units + Dropout 0.2
- **Layer 4:** ANN Dense 64 units (ReLU)
- **Layer 5:** ANN Dense 32 units (ReLU)
- **Output:** 24-hour ahead SoC forecast (%)
- **Battery:** 240kWh Lithium-ion, 92% efficiency, 10-95% SoC range

## 📊 Dataset (2024/2025)
- **File:** `kajola_village_2024_2025_dataset.csv`
- **Samples:** 17,544 hourly samples (Jan 1, 2024 - Dec 31, 2025)
- **Features:** Timestamp, Load_kW (48-210kW), Solar_Irradiance_Wm2 (0-950), Temp_C, Humidity_%, Hour, Day, Month, Solar_Gen_kW, SoC_%
- **Location:** Kajola Village, Sagamu (6.8°N, 3.6°E)

## 📈 Results (6 Figures)
- **Figure 4.1:** Training & Validation Loss Curve - Converged at Epoch 32 (2.1e-04 loss)
- **Figure 4.2:** Weekly Load Demand Profile (48kW-210kW variation)
- **Figure 4.3:** Solar Irradiance Pattern (Dry vs Wet season)
- **Figure 4.4:** Temperature vs Load Correlation Scatter Plot
- **Figure 4.5:** Actual vs Predicted SoC (RMSE 0.39%, MAE 0.26%)
- **Figure 4.6:** 24-Hour Ahead Forecast for 2026/2027 (9.89%-10.03% SoC) - PRIMARY AIM

## 🔮 2026/2027 Forecast Output
- **File:** `forecast_2026_2027_24h.csv`
- **Mean SoC:** 9.96%
- **Available Energy:** 23.9 kWh (9.96% x 240kWh)
- **Insight:** Critical storage depletion - requires enhanced charging strategy

## 🚀 How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
