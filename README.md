# Anomaly Detection Agent

A Streamlit-based internal anomaly detection application that allows users to upload their own datasets, detect potentially anomalous records using classical machine learning methods, and review the most suspicious observations through an interactive interface.

---

## Project Overview

This project is designed for internal company use and enables non-technical or semi-technical users to upload structured datasets and run anomaly detection without needing to build a model from scratch.

In the first phase, the application is built entirely with traditional data processing and machine learning components. Users can upload their own files, preview the data, select relevant columns, run anomaly detection, and inspect the most suspicious records.

The UI is developed with Streamlit to provide a simple and accessible user experience.

A future second phase may introduce an LLM layer for natural language explanations, guided column selection, and conversational interpretation of results. However, the current version does not depend on any LLM.

---

## Goals

- Enable users to upload their own tabular data and analyze it easily  
- Detect unusual or suspicious observations using classical ML algorithms  
- Provide a lightweight and interactive Streamlit interface for internal use  
- Support human review by surfacing the top anomalous records  
- Create a scalable baseline architecture that can later be extended with LLM-powered explanations  

---

## Scope

### Included in this version

- Dataset upload through Streamlit  
- Support for CSV and Excel files  
- Basic data inspection and preview  
- Data preprocessing for anomaly detection  
- Classical ML-based anomaly detection  
- Ranking and displaying anomalous observations  
- Manual review of suspicious rows  

### Not included in this version

- LLM integration  
- Natural language chat interface  
- Automated root-cause explanations  
- Supervised anomaly models  
- Feedback-based learning  

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

1. User uploads dataset  
2. System inspects structure  
3. Relevant columns are selected  
4. Data is preprocessed  
5. Anomaly detection model runs  
6. Each row receives an anomaly score  
7. Top suspicious records are displayed  

---

## Model Approach

This project uses classical anomaly detection techniques.

### Isolation Forest

Isolation Forest is an unsupervised anomaly detection algorithm that identifies unusual observations by isolating them through random partitioning.

Key characteristics:

- Does not require labeled data  
- Works well for tabular datasets  
- Efficient and scalable  
- Effective when anomalies are rare  

Other models (future consideration):

- Local Outlier Factor (LOF)  
- One-Class SVM  
- Statistical threshold methods  

---

## System Architecture

### 1. User Interface Layer
- Built with Streamlit  
- File upload and interaction  
- Visualization of results  

### 2. Data Processing Layer
- File reading  
- Column type detection  
- Missing value handling  
- Encoding categorical variables  

### 3. ML Layer
- Model execution  
- Anomaly scoring  
- Ranking suspicious records  

### 4. Review Layer
- Display top anomalies  
- Enable manual inspection  
- Support decision-making  

---

## Features

- Upload CSV or Excel files  
- Preview datasets  
- Select columns for analysis  
- Run anomaly detection  
- Display anomaly scores  
- Highlight suspicious records  
- Simple Streamlit interface  

---

## Example Use Case

A user uploads an operational dataset. The system processes the data and identifies unusual records based on statistical patterns. The most suspicious entries are displayed, allowing the user to investigate potential issues, rare events, or data inconsistencies.

---

## Tech Stack

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Scikit-learn  
- OpenPyXL / XlsxWriter  

---

## Project Structure
anomaly-detection-agent/
│
├── app/
│ ├── streamlit_app.py
│ ├── logic/
│ │ ├── preprocessing.py
│ │ ├── anomaly_model.py
│ │ └── file_loader.py
│ └── utils/
│
├── data/
│ └── sample_files/
│
├── notebooks/
│ └── experiments.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore


---

## Input Format

Supported formats:

- CSV  
- Excel  

Expected structure:

- Rows = observations  
- Columns = features  
- Mostly structured data  

Users should avoid including:
- ID-only columns  
- irrelevant free-text fields  

---

## Output

The system produces:

- Anomaly scores  
- Anomaly ranking  
- Top suspicious records  

Optional outputs:

- Downloadable results  
- Filtered anomaly view  
- Summary statistics  

---

## Limitations

- No business context awareness  
- Performance varies by dataset  
- Limited support for text-heavy data  
- No natural language explanations  
- No feedback learning yet  

This tool should be used as a **decision support system**, not a fully automated solution.

---

## Future Enhancements

- LLM-based explanation layer  
- Natural language interface  
- Automated feature suggestions  
- Feedback-based model improvement  
- Multiple model support  
- Advanced visualization  
- Domain-specific configurations  

---

## Phase Planning

### Phase 1
- Classical ML anomaly detection  
- Streamlit UI  

### Phase 2
- LLM integration  
- Natural language explanations  
- Conversational interface  

---

## Requirements
streamlit
pandas
numpy
scikit-learn
openpyxl
