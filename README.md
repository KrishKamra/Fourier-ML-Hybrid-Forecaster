# 📈 Nexus Fourier: AI Time-Series Forecaster

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-05998b.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Nexus Fourier** is a hybrid forecasting engine designed for high-precision time-series prediction. It bridges the gap between classical signal processing and modern machine learning by using the **Fast Fourier Transform (FFT)** to augment feature vectors for Gradient Boosted Trees (XGBoost).

---

## 🧠 Core Architecture
The system follows a modular pipeline architecture:
1. **Signal Ingestion:** Generates or fetches raw time-domain signals.
2. **Fourier Extraction:** Performs an RFFT to identify dominant periodicities ($T$).
3. **Feature Augmentation:** Concatenates frequency-domain coefficients with time-domain lags.
4. **ML Inference:** Uses XGBoost to predict the future values based on the "Spectral DNA" of the input window.

## 🛠️ Project Structure
```text
fourier-forecast-ai/
├── api/                # FastAPI implementation for RESTful inference
├── config/             # YAML-based centralized parameter management
├── data_ingestion/     # Synthetic & real-world data generators
├── evaluation/         # Mathematical error metrics (MAE, RMSE, MAPE)
├── feature_engineering/# FFT extraction logic (The Math Core)
├── inference/          # Predictor class for production deployment
├── models/             # XGBoost regressor implementation
├── preprocessing/      # Sliding window & normalization logic
├── tests/              # Unit tests and presentation lab scripts
├── visualization/      # Streamlit SaaS-style Dashboard
└── main.py             # CLI entry point for diagnostic runs
```
## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed. Clone the repository and install dependencies:
```bash
git clone [https://github.com/yourusername/fourier-forecast-ai.git](https://github.com/yourusername/fourier-forecast-ai.git)
cd fourier-forecast-ai
pip install -r requirements.txt

```
### 2. Launch the Backend (API)
The API handles the Fourier-ML inference pipeline.
```bash
uvicorn api.server:app --reload
```
*Access the interactive documentation at http://127.0.0.1:8000/docs*

### 3. Launch the Frontend (Dashboard)
The dashboard provides a SaaS-style UI for spectral analysis and live forecasting.
```bash
python -m streamlit run visualization/dashboard.py
```

## 📊 Presentation Lab
The included **Presentation Lab** module allows users to:
- Generate custom signals with variable periods ($T$).
- Observe the real-time shift in the **Power Spectrum**.
- Verify the **Inverse Relationship** between frequency and time.
- Copy-paste JSON payloads for live API testing via Swagger UI.

## 🧪 Mathematical Validation
By integrating the Fourier Transform, the model achieves a significant reduction in **RMSE** compared to baseline autoregressive models. The FFT allows the model to treat seasonality as a primary feature rather than a residual, leading to faster convergence and better interpretability.

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

---

