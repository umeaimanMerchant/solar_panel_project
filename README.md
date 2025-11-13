# Solar Panel Feasibility Analysis — End-to-End Project

## 📘 Project Overview

The **Solar Footprints Feasibility Analysis in California** project leverages **Machine Learning and Apache Spark** to predict the feasibility of solar installations based on **geographic, environmental, and infrastructural factors**. The aim is to help **developers, utility companies, and policymakers** make informed, data-driven decisions on where to establish solar farms efficiently and sustainably.

---

## ⚙️ Setup

### Prerequisites

* Python 3.12
* Conda or virtual environment
* Git
* DVC (Data Version Control)
* MLflow (for experiment tracking)

### Step-by-Step Installation

```bash
# STEP 01: Clone the repository
git clone https://github.com/<your-username>/Solar-Feasibility-Project.git
cd Solar-Feasibility-Project

# STEP 02: Create a conda environment
conda create -n solar python=3.8 -y
conda activate solar

# STEP 03: Install requirements
pip install -r requirements.txt

# STEP 04: Run the app
python app.py
```

Open your local host and port to access the web interface.

---

## 🧩 Project Structure

```
Solar-Feasibility-Project/
│
├── src/
│   ├── config/
│   │   ├── config.yaml
│   │   ├── params.yaml
│   │   ├── secrets.yaml  # Optional (DB/API credentials)
│   │
│   ├── entity/
│   │   └── data_config.py
│   │
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │
│   ├── pipeline/
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_data_transformation.py
│   │   ├── stage_03_model_training.py
│   │   ├── stage_04_model_prediction.py
│   │
│   ├── utils/
│   └── main.py
│
├── artifacts/
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── app.py
├── setup.py
├── requirements.txt
├── dvc.yaml
└── README.md
```

---

## 🚀 Modular Workflow

### **1. Update YAML Files**

* **config.yaml** → All project paths, URLs, and constants
* **params.yaml** → Model parameters, tuning configs
* **secrets.yaml** → (Optional) DB or API credentials

### **2. Entity Layer**

Defines data classes (entities) that hold structured configurations like paths and dataset info.

### **3. Configuration Manager**

Reads YAML files and creates artifact directories automatically.
It initializes entity classes and passes configuration values dynamically.

### **4. Components**

* **Data Ingestion:** Downloads the dataset (ZIP) and extracts it into artifacts
* **Data Transformation:** Handles preprocessing for both **train** and **test** data
* **Model Training:** Builds and trains models using MLlib or scikit-learn
* **Model Evaluation:** Evaluates model performance and logs metrics

### **5. Pipeline**

Connects all stages sequentially.
The data flows through:
`YAML Config → Config Manager → Entity → Component → Pipeline → Output`

---

## 🔁 Modular Data Ingestion (Simplified Flow)

1. **Create YAML Constants**
   Store configurations like paths, URLs, secrets.

2. **Manage Secrets**
   Securely connect to databases or APIs.

3. **Create Artifact Storage**
   Automatically create structured folders (`artifacts/raw`, `artifacts/processed`).

4. **Configuration Manager**
   Reads YAML → Creates folders → Initializes entities.

5. **Data Configuration Entity**
   Stores ingestion parameters (URLs, formats, etc.).

6. **Data Ingestion Component**
   Downloads and extracts datasets.

7. **Pipeline Execution**
   Orchestrates end-to-end flow for modularity and maintainability.

---

## 📊 MLflow Integration

Track your experiments with MLflow.

```bash
mlflow ui
```

### Example DagsHub Integration

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/Solar-Feasibility-MLflow-DVC.mlflow
export MLFLOW_TRACKING_USERNAME=<your-username>
export MLFLOW_TRACKING_PASSWORD=<your-password>
```

Run:

```bash
python script.py
```

---

## 📦 DVC Commands

```bash
dvc init
dvc repro
dvc dag
```

**Purpose:**

* `dvc repro`: Reproduces full ML pipeline
* `dvc dag`: Shows visual pipeline flow

---

## ☁️ AWS CI/CD Deployment (via GitHub Actions)

### **Steps:**

1. **Login to AWS Console**
2. **Create IAM User**

   * Access Policies:

     * `AmazonEC2ContainerRegistryFullAccess`
     * `AmazonEC2FullAccess`
3. **Create ECR Repository**

   * Example URI: `4376493.dkr.ecr.us-east-1.amazonaws.com/solar-project`
4. **Create EC2 Instance**

   * Install Docker:

     ```bash
     sudo apt-get update -y
     sudo apt-get upgrade
     curl -fsSL https://get.docker.com -o get-docker.sh
     sudo sh get-docker.sh
     sudo usermod -aG docker ubuntu
     newgrp docker
     ```
5. **Setup Self-Hosted Runner** in GitHub
6. **Add GitHub Secrets:**

| Secret Name             | Description     |
| ----------------------- | --------------- |
| `AWS_ACCESS_KEY_ID`     | IAM access key  |
| `AWS_SECRET_ACCESS_KEY` | IAM secret      |
| `AWS_REGION`            | us-east-1       |
| `AWS_ECR_LOGIN_URI`     | AWS ECR URI     |
| `ECR_REPOSITORY_NAME`   | Repository name |

---

## 🧠 Business Context

As California pushes toward renewable energy, **identifying feasible solar installation sites** is vital.
This project enables:

* **Smart site selection** using data-driven insights
* **Cost savings** by avoiding non-feasible locations
* **Accelerated renewable energy adoption** through efficient data processing
* **Sustainability** by aligning investments with environmental goals

---

## 🎯 Main Objectives

1. **Classify Solar Site Feasibility** using Spark MLlib models.
2. **Optimize Resource Allocation** with SparkSQL analytics.
3. **Support Policy and Investment Planning** using big data insights.
4. **Reduce Risk** of failed or non-feasible installations.

---

## 💡 Future Work

* Separate data transformation for **train** and **test** datasets.
* Extend ML pipelines for **real-time prediction APIs**.
* Integrate with **cloud-based data lakes (AWS S3 / GCP Storage)**.
* Visualize feasibility maps using **geospatial dashboards (Plotly, Folium)**.

---

## 🧾 References

* [Kidney Disease Classification (Base Reference Project)](https://github.com/krishnaik06/Kidney-Disease-Classification-Deep-Learning-Project)
* [MLflow Documentation](https://mlflow.org/)
* [DVC Documentation](https://dvc.org/doc)


