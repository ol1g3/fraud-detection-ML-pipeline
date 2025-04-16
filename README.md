# fraud-detection-ML-pipeline
Fraud Detection System using Machine Learning, with a complete CI/CT/CD pipeline

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

## Data Source

The dataset used for this project can be found on Kaggle: [Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection).