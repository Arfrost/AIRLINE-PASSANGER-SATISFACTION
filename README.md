# ✈️ Airline Passenger Satisfaction Prediction

This repository contains a comprehensive data science project aimed at predicting airline passenger satisfaction. Using a dataset of over 100,000 flight records, we explore the factors that lead to a "Satisfied" vs. "Neutral or Dissatisfied" experience through advanced statistical modeling and machine learning.

##  Project Overview
The objective is to identify key drivers of customer loyalty and build a model that accurately classifies passenger sentiment. The workflow includes:

* **Exploratory Data Analysis (EDA):** Visualizing service quality impacts (WiFi, Food, Comfort).
* **Dimensionality Reduction:** Applying **Principal Component Analysis (PCA)** to handle feature multicollinearity, retaining **91.6%** of variance with 12 components.
* **Machine Learning Pipeline:** Comparative analysis of 6 different architectures.
* **Hyperparameter Tuning:** Utilizing `RandomizedSearchCV` for model optimization.

## Final Model Performance
The models are ranked below by their **ROC-AUC** scores. Ensemble methods (XGBoost and Random Forest) demonstrated near-perfect separation capabilities.

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **XGBoost (Optimized)** | **0.9635** | **0.9728** | **0.9428** | **0.9575** | **0.9950** |
| 2 | Random Forest (Optimized) | 0.9621 | 0.9722 | 0.9401 | 0.9559 | 0.9939 |
| 3 | MLP (Neural Network) | 0.9288 | 0.9370 | 0.8971 | 0.9166 | 0.9810 |
| 4 | Decision Tree | 0.9539 | 0.9587 | 0.9347 | 0.9466 | 0.9752 |
| 5 | SVM (Optimized) | 0.9117 | 0.9096 | 0.8856 | 0.8975 | 0.9669 |
| 6 | Logistic Regression (PCA) | 0.8265 | 0.8168 | 0.7765 | 0.7961 | 0.8870 |

##  Key Insights & Methodology

###  Dimensionality Reduction (PCA)
To optimize training for complex models like SVM and MLP, PCA was implemented. By reducing the feature space to 12 components, we maintained high accuracy while significantly reducing computational overhead.

###  Ensemble Dominance
The **XGBoost** model outperformed all others, achieving a **96.35% Accuracy**. This suggests that the relationship between flight features (like *Inflight WiFi* and *Online Boarding*) and satisfaction is highly non-linear, which gradient boosting handles exceptionally well.

###  Optimization Strategy
Each model was fine-tuned using a 3-fold cross-validation `RandomizedSearchCV`. This ensured that the metrics reported are robust and not a result of overfitting to the training set.

##  Tech Stack
* **Language:** Python
* **Data Ops:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`, `xgboost`

##  Getting Started
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/airline-satisfaction-prediction.git](https://github.com/your-username/airline-satisfaction-prediction.git)
