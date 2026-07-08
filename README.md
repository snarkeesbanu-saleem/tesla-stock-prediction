# Tesla Stock Prediction & Quantitative Analytics Platform

A high-performance interactive deep learning simulation and exploratory quantitative analytics platform designed for Tesla (TSLA) stock data. This Python application integrates advanced mathematical models, interactive neural network configurations (LSTM, SimpleRNN), natural language processing indicators, and interactive sentiment indexing.

**🔗 Deployed Live Application**: [tesla-stock-prediction-felnadqvxkiagt3fwxbixe.streamlit.app](https://tesla-stock-prediction-felnadqvxkiagt3fwxbixe.streamlit.app/)

Designed with a premium dark executive design system, the dashboard supports multiple highly appealing, high-contrast dark colorways and fully customized data visualizations using Plotly and Streamlit.

---

## 🚀 Key Architectural Features

### 1. Interactive Neural Network Trainer & Simulator
*   **Model Selection**: Easily toggle and compare neural network configurations such as Long Short-Term Memory (LSTM) and Simple Recurrent Neural Networks (SimpleRNN).
*   **Hyperparameter Customization**: Adjust learning rates, epoch duration, sequence input lengths, activation functions, and neuron counts directly from the UI.
*   **Live Training Visualizer & Motion**: Watch real-time multi-epoch loss minimization curves and dynamic ground-truth comparison charts alongside a live-flickering gate weights profiling heatmap matrix as the simulated model converges.

### 2. Typology-Filtered Quantitative Engine (15 Explanatory Charts)
Isolate metrics and discover deep statistical patterns through an interactive deck grouping exactly **15 quantitative and mathematical charts** cataloged into four distinct analytical pools:
*   **Area & Channel Profiles**: Closed pricing bounds, Moving Average envelopes, and comprehensive Bollinger Band channel bands.
*   **Mathematical & Trend Lines**: Exponential and Simple Moving Averages (EMA20, SMA50, SMA200), MACD momentum crossovers, and Support/Resistance price boundaries.
*   **Frequency & Spread Bars**: Volatility histograms, Absolute return distributions, day-over-day open-close standard deviations, and volume spreads.
*   **Correlation Dispersion**: Custom scatter distributions plotting high-density volume velocity against daily return variance.

### 3. State-of-the-Art Executive Dark Themes
A built-in custom visual theme management system optimized for prolonged structural data analysis under multiple gorgeous black color palettes:
*   **Obsidian Crimson**: High-octane performance Tesla Crimson highlights matching premium charcoal-carbon black accents.
*   **Midnight Amber**: Professional brushed champagne gold accents set over ultra-clean executive velvet-black slate.
*   **Cyber Pulse**: High-voltage cyber cyan and futuristic violet highlights tailored for modern screen displays.
*   **Emerald Matrix**: Tactical mint green cybernetic terminal styling designed for rapid matrix readability.
*   **Classic Paper**: Formal eggshell white and high-contrast charcoal for the clean aesthetic of academic publication prints.

### 4. Advanced Sentiment Analysis & Reporting
*   **Interactive News Timelines**: Navigate through high-impact Tesla macro news events synced across all charts and metrics.
*   **Analyst Sentiment Analyzer**: Evaluate natural language processing scoring indices correlating physical sentiment shifts to actual stock volatility using direct Gemini REST API integration.
*   **Comprehensive Financial Reports**: Generated analytical digests providing situational risk, technical layouts, and target price brackets.

---

## 🛠️ Technology Stack

*   **Framework**: Streamlit (v1.38+)
*   **Data Science**: Pandas, NumPy
*   **Data Visualizations**: Plotly (v5.24+)
*   **HTTP Clients**: Requests (for calling Gemini REST API)
*   **Environment**: Python (v3.12+)

---

## 📂 Project Structure

```bash
├── assets/             # Branding and tech banner assets
│   └── tesla_dashboard_hero.jpg
├── app.py              # Main Streamlit application and layout builder
├── models.py           # Custom SimpleRNN and LSTM network propagation logic
├── stock_data.py       # TSLA dataset loader, indicator computations, and news logs
├── requirements.txt    # Python package dependencies
└── README.md           # Documentation and guidelines
```

---

## 💻 Setup and Installation

Follow these steps to run the application locally on your computer:

1.  **Navigate to the Project Directory**:
    ```bash
    cd tesla-stock-prediction
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit Dashboard**:
    ```bash
    streamlit run app.py
    ```
    Access the application in your browser at `http://localhost:8501` (or the local port provided in your terminal).
