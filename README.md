# Anomaly Detection Agent

A Streamlit-based internal anomaly detection application that allows users to upload their own datasets, detect potentially anomalous records using classical machine learning methods, and review the most suspicious observations through an interactive interface.

---

## Project Overview

This project is designed for internal company use and enables non-technical or semi-technical users to upload structured datasets and run anomaly detection without needing to build a model from scratch.

The application is built entirely with traditional data processing and machine learning components. Users can upload their own files, preview the data, select relevant columns, run anomaly detection with automatic model selection, inspect the most suspicious records, and provide feedback to improve results.

The UI is developed with Streamlit to provide a simple and accessible user experience. All interface elements are designed for users without technical ML knowledge.

A future second phase may introduce an LLM layer for natural language explanations, guided column selection, and conversational interpretation of results. However, the current version does not depend on any LLM.

---

## Goals

- Enable users to upload their own tabular data and analyze it easily
- Detect unusual or suspicious observations using classical ML algorithms
- Automatically select the best model for the given dataset
- Provide a lightweight and interactive Streamlit interface for internal use
- Support human review by surfacing the top anomalous records
- Allow users to provide feedback and improve detection with semi-supervised learning
- Create a scalable baseline architecture that can later be extended with LLM-powered explanations

---

## Scope

### Included in this version

- Dataset upload through Streamlit (CSV, Excel, ZIP)
- Maximum upload size: 1 GB
- Basic data inspection and preview
- Automatic column quality analysis and recommendations
- ID column detection and identifier selection
- Data preprocessing with OneHotEncoding and frequency encoding
- Three classical ML models with automatic benchmarking and selection
- Time series anomaly detection for datasets with date columns
- Ranking and displaying anomalous observations
- Feature-level deviation explanations for each suspicious record
- Human feedback loop (true anomaly / false alarm)
- Semi-supervised retraining based on user feedback
- Excel export with neon yellow highlighting on anomaly rows
- Preset scenario configurations for common dataset types
- Manual review of suspicious rows

### Not included in this version

- LLM integration
- Natural language chat interface
- Automated root-cause explanations
- Supervised anomaly models
- Feedback persistence across sessions

---

## Why Classical Machine Learning?

Anomaly detection is often a better fit for classical ML than LLM-based architectures, especially in early stages.

Advantages:

- Lower complexity
- Faster implementation
- Easier deployment in internal environments
- Lower operational cost
- More predictable behavior
- No dependency on external AI services

---

## Core Workflow

1. User uploads dataset (CSV, Excel, or ZIP)
2. System inspects structure and previews data
3. Column quality analysis runs automatically (ID-like, constant, free text, high cardinality)
4. User selects identifier columns (shown as reference in results, excluded from analysis)
5. User selects analysis columns (safe defaults pre-selected, flagged columns softly warned)
6. Three anomaly detection models run and are benchmarked automatically
7. Best model is selected based on silhouette score
8. Each row receives an anomaly score and is ranked
9. Top suspicious records are displayed with feature-level explanations
10. Optional: Time series anomaly detection if date columns are present
11. User provides feedback (true anomaly / false alarm)
12. Optional: Semi-supervised retraining with feedback
13. Results are downloaded as Excel with highlighted anomaly rows

---

## Model Approach

This project uses three classical anomaly detection techniques. All three models are run automatically and the best one is selected based on silhouette score.

### Isolation Forest

Isolates anomalies through random partitioning. Works well for general-purpose tabular data.

- Does not require labeled data
- Efficient and scalable
- Effective when anomalies are rare

### Local Outlier Factor (LOF)

Compares each record's local density to its neighbors. Good at finding cluster-based outliers.

- Density-based approach
- Effective for datasets with varying cluster densities
- Recommended for HR and similar grouped datasets

### One-Class SVM

Draws a boundary around normal data and flags records outside it.

- Kernel-based (RBF)
- Effective for well-separated anomalies
- Higher computational cost on large datasets

### Temporal Anomaly Detection

For datasets with date columns, a rolling statistics approach is available.

- Rolling mean and standard deviation with configurable window
- Flags data points outside rolling_mean +/- 2.5 * rolling_std
- Visualizes trend, confidence band, and anomalous time points

### Semi-Supervised Retraining

After user feedback, the model is retrained:

- Records marked as "normal" are included in the training set
- Records marked as "anomaly" are excluded from training
- Updated results are displayed and downloadable

---

## System Architecture

### 1. User Interface Layer
- Built with Streamlit
- File upload (CSV, Excel, ZIP)
- Interactive column selection with quality recommendations
- ID column selection for result referencing
- Preset scenario tips
- Visualization of results with Plotly
- Feedback buttons for each suspicious record
- Excel download with styled anomaly highlighting

### 2. Data Processing Layer
- File reading (CSV, Excel, ZIP with automatic extraction)
- Column type detection (numeric, categorical, datetime)
- Column quality analysis (ID-like, constant, free text, high cardinality detection)
- Missing value handling (median for numeric, mode for categorical)
- OneHotEncoding for low-cardinality categoricals (<=15 categories)
- Frequency encoding for high-cardinality categoricals (>15 categories)
- StandardScaler normalization

### 3. ML Layer
- Automatic model benchmarking (Isolation Forest, LOF, One-Class SVM)
- Model selection via silhouette score
- Anomaly scoring and ranking
- Feature deviation analysis per record
- Temporal anomaly detection with rolling statistics
- Semi-supervised retraining with user feedback

### 4. Review Layer
- Display top anomalies with ID columns as reference
- Feature-level explanations (value vs. typical, deviation magnitude)
- Human feedback collection
- Updated results after feedback-based retraining

---

## Features

- Upload CSV, Excel, or ZIP files (up to 1 GB)
- Preview datasets with summary statistics
- Automatic column quality analysis and soft recommendations
- ID column detection and identifier selection
- Preset scenario tips (General, Finance, HR, Call Center)
- Automatic model benchmarking and selection
- Display anomaly scores and suspicious record rankings
- Feature-level deviation explanations for each record
- Time series anomaly detection with trend visualization
- Human feedback loop (true anomaly / false alarm)
- Semi-supervised retraining based on feedback
- Download results as Excel with neon yellow highlighted anomaly rows
- Non-technical Turkish UI

---

## Preset Scenarios

The application provides preset tips for common dataset types. These do not affect the pipeline; they serve as guidance for the user.

| Preset | Description | Suggested Contamination |
|--------|-------------|------------------------|
| Genel (Varsayilan) | General-purpose datasets | 5% |
| Islem / Finans Verisi | Payments, invoices, insurance claims | 2% |
| Insan Kaynaklari Verisi | Salary, performance, attendance | 5% |
| Cagri Merkezi Verisi | Call duration, wait time, satisfaction | 5% |

---

## Example Use Case

A user uploads an operational dataset. The system automatically analyzes column quality, detects ID columns, and recommends safe columns for analysis. Three anomaly detection models run in parallel and the best one is selected. The most suspicious entries are displayed with explanations showing which features deviate most from typical values. The user reviews results, marks some as true anomalies or false alarms, and reruns the model with feedback. Final results are downloaded as an Excel file with anomaly rows highlighted in yellow.

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib
- OpenPyXL

---

## Project Structure

```
anomaly-detection/
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py          # Main Streamlit application
│   ├── logic/
│   │   ├── __init__.py
│   │   ├── file_loader.py        # CSV, Excel, ZIP file loading
│   │   ├── preprocessing.py      # Column analysis, encoding, scaling
│   │   ├── anomaly_model.py      # ML models, benchmarking, temporal analysis
│   │   └── presets.py            # Preset scenario configurations
│   └── utils/
│       └── __init__.py
├── data/
│   └── sample_files/
├── .streamlit/
│   └── config.toml               # Theme and upload size configuration
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Input Format

Supported formats:

- CSV
- Excel (.xls, .xlsx)
- ZIP (containing CSV or Excel files)

Expected structure:

- Rows = observations
- Columns = features
- Mostly structured data

The system automatically detects and warns about:
- ID-only columns (high unique ratio or name contains "id")
- Constant columns (single value)
- Free-text fields (high average word count)
- High-cardinality categoricals (>50 unique values)

Users can still include warned columns if they choose to.

---

## Output

The system produces:

- Anomaly scores for each record
- Anomaly ranking (most suspicious first)
- Top suspicious records with ID columns as reference
- Feature-level deviation explanations
- Time series anomaly visualization (if applicable)
- Feedback summary statistics

Downloadable outputs:

- Full results as Excel (.xlsx) with anomaly rows highlighted in neon yellow
- Top N suspicious records as Excel (.xlsx) with highlighting
- Updated results after semi-supervised retraining

---

## Limitations

- No business context awareness
- Performance varies by dataset
- Limited support for text-heavy data
- No natural language explanations
- Feedback is not persisted across sessions
- Semi-supervised retraining uses only Isolation Forest

This tool should be used as a **decision support system**, not a fully automated solution.

---

## Future Enhancements

- LLM-based explanation layer
- Natural language interface
- Automated feature suggestions
- Feedback persistence and model versioning
- Ensemble model support
- Advanced visualization
- Domain-specific configurations
- API endpoints for integration

---

## Phase Planning

### Phase 1 (Current)
- Classical ML anomaly detection (Isolation Forest, LOF, One-Class SVM)
- Automatic model benchmarking and selection
- Column quality analysis and ID detection
- Time series anomaly detection
- Human feedback loop with semi-supervised retraining
- Excel export with anomaly highlighting
- Streamlit UI with preset scenario tips

### Phase 2
- LLM integration
- Natural language explanations
- Conversational interface
- Feedback persistence

---

## Deployment

This application is designed for Streamlit Cloud deployment.

1. Push the repository to GitHub
2. Connect to Streamlit Cloud
3. Set the main file path to `app/streamlit_app.py`

To run locally:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Requirements

```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
openpyxl>=3.1.0
plotly>=5.18.0
matplotlib>=3.7.0
```
