# Prescription Uptake Early-Warning System (PEUS)

An end-to-end analytics platform that detects abnormal changes in prescription volumes, quantifies business impact, and prioritizes actions through SQL-backed data processing, time-series forecasting, and comprehensive validation metrics.

## Overview

PEUS is a proof-of-concept system designed to catch prescription uptake anomalies early—before they impact revenue. By combining Prophet-based forecasting with robust anomaly detection, the platform identifies meaningful deviations in real-time and surfaces actionable insights to commercial teams via an interactive Streamlit dashboard.

### Key Capabilities

- **Early Anomaly Detection**: Identifies significant drops and spikes in prescription volumes using MAD-based z-scores
- **Business Impact Quantification**: Estimates revenue-at-risk and severity classification (Low / Medium / High)
- **Model Validation**: Comprehensive backtesting with precision, recall, and detection latency metrics
- **Executive Dashboard**: Real-time visualization of anomalies, impact scores, and system performance
- **Explainability**: Clear, interpretable cause inference for each detected anomaly

## Why It Matters

Commercial teams traditionally rely on monthly reports to identify issues like supply disruptions, competitor actions, or failed promotions. By then, the damage is done. PEUS detects these changes within days, enabling faster, data-driven decision-making and minimizing revenue loss.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.9+ |
| **Database** | MySQL |
| **Data Processing** | pandas, NumPy |
| **Forecasting** | Prophet |
| **Visualization** | Streamlit |
| **Config Management** | python-dotenv |

## Architecture

```
Synthetic Data (CSV)
        ↓
[Data Ingest & Validation]
        ↓
MySQL Database (Source of Truth)
        ↓
[Time-Series Baseline - Prophet]
        ↓
[Anomaly Detection Engine]
        ↓
[Impact Scoring Module]
        ↓
[Backtesting & Metrics]
        ↓
Streamlit Dashboard (UI)
```

## Project Structure

```
src/
├── utils/           # Synthetic data generation
├── ingest/          # Data validation & loading pipeline
├── db/              # MySQL configuration & utilities
├── models/          # Core ML modules
│   ├── anomaly_detector.py
│   ├── impact_scorer.py
│   └── backtesting.py
└── dashboard/       # Streamlit web application
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- MySQL server running
- Environment variables configured (see `.env` setup)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your MySQL credentials
```

### Running the System

```bash
# Step 1: Generate synthetic data
python3 src/utils/synth_data.py

# Step 2: Ingest and validate data into MySQL
python3 -m src.ingest.load_data

# Step 3: Launch the dashboard
PYTHONPATH=. streamlit run src/dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`.

## Model Performance

The system validates anomaly detection accuracy using backtesting metrics:

| Metric | Description |
|--------|-------------|
| **Precision** | Proportion of detected anomalies that are true positives |
| **Recall** | Proportion of true anomalies that were detected |
| **False Positive Rate** | Percentage of normal periods flagged as anomalies |
| **Detection Latency** | Average delay (in weeks) from anomaly onset to detection |

All metrics are available in the dashboard's validation panel.

## Configuration

Key settings are managed via environment variables in `.env`:

```env
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=peus_db
```

## Features in Detail

### Anomaly Detection
- MAD (Median Absolute Deviation) based z-score normalization
- Robust to outliers and seasonal patterns
- Configurable sensitivity thresholds

### Impact Scoring
- Revenue-at-risk estimation based on baseline trends
- Severity classification for prioritization
- Explainable change attribution

### Backtesting Framework
- Ground-truth validation on synthetic events
- Performance metrics dashboard
- Iterative model tuning support

## Notes & Limitations

- **Data**: Currently uses synthetic data for demonstration purposes
- **Production Readiness**: Designed for clarity and explainability; adapt for production workloads as needed
- **Extensibility**: Framework supports integration with real prescription data and external data sources

## Future Roadmap

- [ ] Real production data integration
- [ ] Advanced forecasting models (ARIMA, ensemble methods)
- [ ] Multi-level alerting system
- [ ] Automated remediation suggestions
- [ ] API endpoint for programmatic access

## Questions?

For issues, suggestions, or contributions, please open an issue or submit a pull request.