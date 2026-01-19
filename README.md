# Flight Delay Prediction at Scale

A scalable distributed machine learning pipeline that predicts flight departure status (Early, On-Time, Delayed) by processing 28 million flight records with weather data on Apache Spark/Databricks. The system achieves 54.6% F1 score using a 6-layer MLP with probability recalibration, demonstrating how multimodal data and deep learning can support airline scheduling decisions and improve passenger experience for 2.9M+ daily passengers.

---

## Problem and Goal

- **Problem:** Flight delays cascade through airline networks, disrupting 2.9 million daily passengers and costing airlines billions annually. Existing prediction systems struggle with severe class imbalance (57% Early, 25% On-Time, 18% Delayed) and fail to leverage multimodal data (flight operations + weather) at scale. Traditional approaches don't account for temporal leakage in time-series data, leading to overoptimistic performance estimates that fail in production.

- **Why It Matters:** Accurate departure predictions enable proactive resource allocation, crew scheduling, and passenger notifications, reducing cascading delays across connected networks. Airlines need production-grade pipelines that process massive datasets efficiently while maintaining prediction integrity through proper cross-validation and class balance handling.

- **Goal:** Build a scalable, leakage-resistant ML pipeline on Databricks that processes 28M flight records (2015-2019) from U.S. DOT and NOAA datasets, compare model families (logistic regression, random forest, MLP) with multiple resampling strategies, and deliver actionable predictions with balanced performance across all three departure classes within a 14-week academic timeline.

---

## Approach

![Pipeline Architecture](Phase%203/Pipeline%20Implementation%20Flow%20Chart.png)

1. **Distributed Data Ingestion & Preprocessing:** Built PySpark ETL pipeline on Databricks (5-10 workers, 160-320GB RAM per worker) to join 28M flight records from U.S. DOT with 131.9M hourly weather observations from NOAA, handling missing values (9.65% dropped), deduplicating records, and creating the OTPW (On-Time Performance & Weather) dataset with temporal alignment.

2. **Advanced Feature Engineering:** Engineered 221 features including cyclic sine/cosine encoding for temporal periodicity (hour, day, month), target encoding for high-cardinality categories (7 features: Tail Number, Origin Airport), PageRank centrality scores for 350+ airports using directed weighted flight network (NetworkX), and decoded HourlySkyConditions into cloud coverage metrics.

3. **Leakage-Resistant Cross-Validation:** Implemented blocked time-series cross-validation with 5 non-overlapping folds where fold-specific scaling, target encoding, and resampling transformations are fit only on training data and applied to validation data, preventing temporal leakage and ensuring production-realistic performance estimates.

4. **Hyperparameter Optimization Benchmarking:** Compared Grid Search, Random Search, and Bayesian Optimization (Optuna) on Logistic Regression; selected parallelized Grid Search after achieving ~3× speedup (16 minutes vs. 500 minutes for Bayesian) while maintaining comparable performance through Databricks' distributed execution.

5. **Multi-Model Experimentation:** Trained 9 model configurations across three families—Logistic Regression with ElasticNet regularization, Random Forest with Gini impurity, and 3 MLP architectures (5-layer, 6-layer, 7-layer)—each with three resampling strategies (SMOTE, oversampling, undersampling) to handle 57%/25%/18% class imbalance.

6. **Probability Recalibration:** Developed post-processing recalibration formula to correct oversampling bias by adjusting predicted class probabilities to match true class distribution (57%/25%/18%), improving F1 by ~6% across all models and aligning predictions with real-world deployment scenarios.

7. **Production Pipeline Orchestration:** Designed modular PySpark pipeline with configurable components (DataPreprocessor, FeatureEngineer, CustomCrossValidator, ResamplingHandler) enabling end-to-end execution from raw data to trained model in 17-30 hours depending on model complexity, with checkpointing and error handling for fault tolerance.

---

## Results

### Technical Deliverables

- **Distributed ML Pipeline:** Production-ready PySpark pipeline processing 28M records (2015-2019) with 221 engineered features across 5 time-series folds, deployed on Databricks cluster (5-10 workers, 128GB driver, 32 cores) with 17-30 hour end-to-end runtime.

- **Best Model - 6-Layer MLP:** Architecture (189→100→50→25→10→3) with oversampling achieved **54.6% F1 score** and 58.69% accuracy on 2019 test data, with class-specific performance: Early (Precision 66.2%, Recall 80.3%), On-Time (Precision 41.6%, Recall 18.7%), Delayed (Precision 44.2%, Recall 30.9%).

- **Recalibration System:** Probability adjustment formula improved F1 by ~6% across 9 model configurations by correcting oversampling bias and aligning predicted class distributions to true rates (57%/25%/18%), critical for production deployment accuracy.

- **Hyperparameter Tuning Framework:** Grid Search with Spark parallelization achieved 3× faster runtime (16 min vs. 500 min) compared to Bayesian Optimization while maintaining model performance, enabling efficient exploration of 100+ hyperparameter combinations.

- **Network-Based Features:** PageRank centrality scores for 350+ airports using directed flight network (5.5K edges) captured hub importance and connectivity, contributing to improved predictions for high-traffic origin airports.

### Key Outcomes

- **Balanced Class Performance:** Delivered actionable minority-class detection (30.9% recall for Delayed, 18.7% for On-Time) while maintaining strong majority-class performance (80.3% recall for Early), addressing severe 57%/25%/18% imbalance through ensemble resampling strategies.

- **Temporal Leakage Prevention:** Blocked time-series cross-validation with fold-specific transforms eliminated data leakage, reducing overfitting and providing production-realistic performance estimates validated against held-out 2019 test data.

- **Scalable Architecture:** Modular PySpark design with configurable components enables easy integration of new features, resampling methods, or models; fault-tolerant execution with checkpointing supports datasets exceeding 100M records.

- **Academic Rigor:** Completed comprehensive 14-week capstone including 30+ experiments across 3 model families, documented in 308KB Phase 3 report with full methodology, ablation studies, confusion matrices, and reproducible Databricks notebooks.

---

## Tech/Methods

**Languages & Frameworks:** Python, PySpark (Apache Spark 3.x), Databricks ML Runtime 15.4, Scikit-learn, TensorFlow/Keras

**Tools & Services:** Databricks (5-10 workers, 160-320GB RAM/worker), NetworkX (graph analysis), Optuna (Bayesian optimization), SMOTE (imbalanced-learn), Pandas, NumPy

**Infrastructure & Data:** AWS Databricks cluster (128GB driver, 32 cores), U.S. DOT Bureau of Transportation Statistics (14.8M flight records), NOAA Integrated Surface Database (131.9M hourly weather observations), Databricks File System (DBFS)

**Methods:** Multi-class classification, Blocked time-series cross-validation, Feature engineering (cyclic encoding, target encoding, PageRank centrality), Hyperparameter tuning (Grid/Random/Bayesian), Class imbalance handling (SMOTE/over/undersampling), Probability recalibration, Distributed computing (MapReduce), Neural networks (MLP), Ensemble methods (Random Forest), Regularization (ElasticNet)

---

## Repo Structure

```
flight-delay-prediction/
├── Phase 1/                        # EDA & Problem Definition
│   └── Phase 1 Report.html         # Exploratory analysis, data quality assessment
│
├── Phase 2/                        # Feature Engineering & Baseline Models
│   ├── Phase 2 Report.html         # OTPW dataset construction, initial models
│   ├── Phase 2 Presentation.pdf    # Milestone presentation slides
│   └── Phase 2 Presentation.pptx   # Presentation source
│
├── Phase 3/                        # Deep Learning & Scale-Up
│   ├── Phase 3 Report.html         # Comprehensive final report (MAIN DOCUMENT)
│   ├── Phase 3 Report.ipynb        # Notebook version of final report
│   ├── Phase 3 - ML Pipeline.ipynb # Production pipeline implementation (ENTRY POINT)
│   ├── Pipeline Implementation Flow Chart.png  # Architecture diagram
│   ├── Phase 3 Presentation.pdf    # Final presentation slides
│   └── Phase 3 Presentation.pptx   # Presentation source
│
└── README.md                       # Project documentation
```

---

## Prerequisites

**Platforms & Services:**

- **[Databricks](https://databricks.com/)** - ML Runtime 15.4+ with Apache Spark 3.x support
  - Cluster configuration: 5-10 workers (160-320GB RAM each), 128GB driver with 32 cores
  - Access to DBFS (Databricks File System) for data storage
- **Data Sources:**
  - [U.S. DOT Bureau of Transportation Statistics](https://www.transtats.bts.gov/) - Flight on-time performance data (2015-2019)
  - [NOAA Integrated Surface Database](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) - Hourly weather observations

**Required Python Packages:**

```bash
pyspark>=3.3.0
scikit-learn>=1.2.0
tensorflow>=2.12.0  # For MLP models
imbalanced-learn>=0.10.0  # For SMOTE
pandas>=1.5.0
numpy>=1.23.0
networkx>=3.0  # For PageRank feature engineering
optuna>=3.1.0  # For Bayesian optimization
matplotlib>=3.6.0
seaborn>=0.12.0
```

### Databricks Cluster Configuration

**Recommended cluster settings:**

```python
# Driver
- Memory: 128 GB
- Cores: 32

# Workers
- Count: 5-10 nodes (scale based on dataset size)
- Memory per worker: 160-320 GB
- Databricks Runtime: ML 15.4 LTS or higher
```

---

## How to Run

### Phase 1: Exploratory Data Analysis (Reference Only)

**Location:** `Phase 1/Phase 1 Report.html`

This phase documents initial data exploration, missing value analysis, and class distribution assessment. Review the HTML report to understand data characteristics and problem formulation. No code execution required.

### Phase 2: Feature Engineering & Baseline Models (Reference Only)

**Location:** `Phase 2/Phase 2 Report.html`

This phase covers OTPW dataset construction, feature engineering methodology, and baseline model experiments (Decision Trees, Logistic Regression, Random Forest). Review the HTML report for feature engineering decisions and initial results. No code execution required.

### Phase 3: Production Pipeline & Deep Learning (MAIN EXECUTION)

**Location:** `Phase 3/Phase 3 - ML Pipeline.ipynb`

This is the **primary entry point** for running the complete production pipeline.

**Step 1: Upload Data to Databricks**

```bash
# Upload raw data files to DBFS
dbfs cp /local/path/to/flights_2015_2019.parquet dbfs:/FileStore/flight_delay_project/raw/flights/
dbfs cp /local/path/to/weather_2015_2019.parquet dbfs:/FileStore/flight_delay_project/raw/weather/
dbfs cp /local/path/to/stations.parquet dbfs:/FileStore/flight_delay_project/raw/stations/
```

**Step 2: Configure Pipeline Parameters**

Open `Phase 3 - ML Pipeline.ipynb` and configure:

```python
# Data paths
FLIGHT_DATA_PATH = "dbfs:/FileStore/flight_delay_project/raw/flights/"
WEATHER_DATA_PATH = "dbfs:/FileStore/flight_delay_project/raw/weather/"
STATIONS_DATA_PATH = "dbfs:/FileStore/flight_delay_project/raw/stations/"

# Model selection
MODEL_TYPE = "mlp"  # Options: "logistic", "random_forest", "mlp"
MLP_ARCHITECTURE = "model_1"  # Options: "model_1" (6-layer), "model_2" (7-layer), "model_3" (5-layer)

# Resampling strategy
RESAMPLING_METHOD = "oversample"  # Options: "smote", "oversample", "undersample"

# Cross-validation
N_FOLDS = 5
```

**Step 3: Run Pipeline**

Execute cells sequentially in `Phase 3 - ML Pipeline.ipynb`:

1. **Data Ingestion & Preprocessing** (~3-5 hours)
   - Loads raw Parquet files into Spark DataFrames
   - Joins flights, weather, and stations on airport codes and timestamps
   - Handles missing values and deduplication
   - Outputs: OTPW dataset with 28M records

2. **Feature Engineering** (~2-4 hours)
   - Creates 221 features: cyclic encoding, target encoding, PageRank centrality
   - Applies fold-specific transformations during cross-validation
   - Outputs: Feature-engineered dataset ready for modeling

3. **Time-Based Cross-Validation** (~1-2 hours)
   - Creates 5 blocked time-series folds (2015-2018 training, 2019 test)
   - Applies fold-specific scaling and encoding to prevent leakage
   - Outputs: Train/validation splits for each fold

4. **Model Training & Evaluation** (~5-20 hours depending on model)
   - Trains selected model with configured resampling strategy
   - Evaluates on each fold and computes aggregate metrics
   - Logistic Regression: ~5 hours | Random Forest: ~8 hours | MLP: ~20 hours
   - Outputs: Trained model, confusion matrices, classification reports

5. **Probability Recalibration** (~30 minutes)
   - Applies recalibration formula to adjust class probabilities
   - Aligns predictions to true class distribution (57%/25%/18%)
   - Outputs: Recalibrated predictions with improved F1 scores

**Expected Runtime:** 17-30 hours total (varies by model complexity and cluster size)

### Quick Start (Using Pre-Trained Results)

If you want to review results without full pipeline execution:

1. Open `Phase 3/Phase 3 Report.html` in a web browser
2. Review comprehensive results including:
   - Confusion matrices for all 9 model configurations
   - Performance metrics (F1, accuracy, precision, recall by class)
   - Hyperparameter tuning comparisons
   - Feature importance analysis
   - Recalibration impact analysis

---

## Notes: Limitations and Next Steps

### Current Limitations

- **Class Imbalance:** Despite resampling and recalibration, minority classes (On-Time: 18.7% recall, Delayed: 30.9% recall) remain harder to predict than the majority Early class (80.3% recall) due to severe 57%/25%/18% distribution skew in real-world data.

- **Feature Lag Constraint:** Model uses only pre-flight features (scheduled times, historical weather) without real-time operational data (gate assignments, crew availability, maintenance status), limiting predictive power for last-minute delays.

- **Temporal Scope:** Limited to 2015-2019 data; does not account for post-COVID operational changes, airline consolidation, or evolving weather patterns due to climate change.

- **Computational Cost:** Full pipeline requires 17-30 hours on a 5-10 node Databricks cluster; MLP training (~20 hours) limits rapid iteration for hyperparameter tuning and architecture search.

- **Airport Coverage:** PageRank features cover only 350+ U.S. airports in the dataset; international connections and smaller regional airports not included in network analysis.

- **Static Model:** Trained once on historical data; requires retraining to adapt to evolving airline schedules, new routes, or changing weather patterns (no online learning).

### Next Steps

- **Real-Time Data Integration:** Incorporate live operational data streams (gate assignments, crew schedules, inbound flight status, maintenance logs) via Apache Kafka or Databricks Delta Live Tables to capture last-minute delay signals and improve minority-class recall by 10-15%.

- **Sequence Modeling:** Replace feature-based MLP with LSTM or Transformer architecture to model temporal dependencies in flight sequences (e.g., cascading delays from inbound flights), leveraging PySpark's distributed deep learning libraries.

- **Cost-Sensitive Learning:** Implement asymmetric loss functions that penalize false negatives for Delayed predictions more heavily than false positives, aligning model optimization with business costs ($75 per delayed passenger vs. $10 per unnecessary alert).

- **Expand Dataset:** Add international flights from FlightAware API, airport operational data (runway configurations, taxiway congestion), and climate forecasts (ECMWF, NOAA GFS) to test generalizability and improve accuracy by 5-8%.

- **Model Compression:** Distill 6-layer MLP into smaller model (3-layer) using knowledge distillation to reduce inference latency from 200ms to <50ms for real-time prediction APIs, enabling deployment on edge devices at airport operations centers.

- **Deploy Production API:** Containerize pipeline with Docker, deploy on Databricks Model Serving or AWS SageMaker with auto-scaling (2-10 instances), and expose REST API with <100ms P95 latency for integration with airline scheduling systems and passenger mobile apps.

---

## Credits / Data / Licenses

### Data Sources

- **U.S. DOT Bureau of Transportation Statistics:** Flight on-time performance data (2015-2019) with 14.8M records covering U.S. domestic flights. Available under public domain for research and educational use. [Access here](https://www.transtats.bts.gov/)

- **NOAA Integrated Surface Database (ISD):** Hourly weather observations (2015-2019) with 131.9M records from 5,000+ surface stations. Available under NOAA's data use policy for non-commercial research. [Access here](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database)

### Frameworks and Tools

- **Apache Spark (PySpark):** Apache License 2.0
- **Scikit-learn:** BSD 3-Clause License
- **TensorFlow/Keras:** Apache License 2.0
- **Databricks ML Runtime:** Commercial platform with free community edition available
- **NetworkX:** BSD 3-Clause License
- **Optuna:** MIT License

### Academic Context

- **Institution:** UC Berkeley School of Information
- **Course:** DATASCI 261 - Machine Learning at Scale
- **Project Duration:** January 2025 - May 2025 (14 weeks)
- **Project Type:** Team capstone project (6 members)

---

## Team Members

| Name | Email | LinkedIn |
|------|-------|----------|
| Kent Bourgoing | kent1bp@berkeley.edu | [LinkedIn](https://www.linkedin.com/in/kentbourgoing/) |
| Sebastian Rosales | (email) | [LinkedIn](https://www.linkedin.com/in/s-rosales/) |
| Kenneth Hahn | (email) | [LinkedIn](https://www.linkedin.com/in/kenneth-hahn-ab981a149/) |
| Benjamin He | (email) | [LinkedIn](https://www.linkedin.com/in/ben-c-he/) |
| Edgar Munoz | (email) | [LinkedIn](https://www.linkedin.com/in/edgar-munoz256/) |
| Adam Perez | (email) | [LinkedIn](https://www.linkedin.com/in/adamperez-datascience/) |

**Phase Leadership:**
- Phase 1 (EDA): Kent Bourgoing & Benjamin He
- Phase 2 (Feature Engineering): Sebastian Rosales & Adam Perez
- Phase 3 (Deep Learning): Kenneth Hahn & Edgar Munoz
