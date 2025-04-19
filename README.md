# fraud-detection-ML-pipeline
A comprehensive end-to-end machine learning pipeline for credit card fraud detection with high accuracy (~99.6%). This project includes the complete workflow from data exploration to model deployment with an integrated CI/CD pipeline.

## Project Overview

This project implements a machine learning system to detect fraudulent credit card transactions. It uses a custom-built fraud detection model trained on transaction data that includes temporal patterns, merchant information, and customer demographics. 

Key features of this pipeline:
- **Data preprocessing** with custom transformations for categorical and temporal features
- **Exploratory data analysis** revealing insights into fraud patterns
- **Model training and evaluation** with high performance metrics
- **Complete CI pipeline** for automated testing and deployment
- **Comprehensive logging system** for tracking model performance and issues
- **Visualization tools** for understanding model decisions and fraud patterns


## Key Insights from Data Analysis

Our exploratory data analysis revealed several important patterns:

1. **Age-Based Vulnerability**: People over 50 years old tend to be more vulnerable to fraud compared to younger age groups.

2. **Temporal Patterns**: Fraud rates vary significantly by hour of day and day of week, with late night - early morning showing higher risk.

3. **Geographic Hotspots**: Large cities like Washington, New York, and Los Angeles have the highest number of fraudulent transactions.

4. **Transaction Categories**: Shopping and groceries are the transaction types showing higher fraud rates than others.

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/ol1g3/fraud-detection-ML-pipeline.git
    cd fraud-detection-ML-pipeline
    ```
2. **Set Up the Virtual Environment:** Choose one of the methods below.

## Virtual Environment Setup Options

### Using venv (default)

1. **Create the Virtual Environment (Mac):**
    ```bash
    python3 -m venv .venv
    ```
2. **Activate the Virtual Environment:**
    ```bash
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
3. **Deactivate the Virtual Environment (when done):**
    ```bash
    deactivate
    ```

### Using uv (alternative)

1. **Create the Virtual Environment:**
    ```bash
    uv venv --python 3.11
    ```
2. **Activate the Virtual Environment And Install Requirements:**
    ```bash
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```
3. **Deactivate the Virtual Environment (when done):**
    ```bash
    deactivate
    ```

## Automated CI Pipeline

This project includes a complete CI/CT pipeline that:

- Runs automated tests on every commit, including build, unit tests
- Validates model performance on test data (regression test)
- Generates performance reports and logs

## Future improvements

- Implement neural network-based approaches for potentially higher accuracy
- Add more sophisticated feature engineering based on domain knowledge


## Data Source

The dataset used for this project can be found on Kaggle: [Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection).