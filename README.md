# Amazon-Sales-Performance-Predictive-Modeling-for-Customer-Lifetime-Value-CLV-


PROJECT OVERVIEW

This project develops a predictive model for Customer Lifetime Value (CLV) using Amazon sales data from 2019 to 2024. CLV represents the total revenue a business can expect from a customer over their relationship with the company. By utilizing historical purchasing patterns—specifically recency, frequency, and monetary (RFM) metrics, this project aims to provide actionable insights for:

A. Optimizing targeted marketing strategies.

B. Enhancing customer retention initiatives.

C. Supporting data-driven business decisions.

The predictive model employs machine learning techniques, with a focus on regression algorithms, to forecast CLV based on customer behavior. The final model, a tuned Random Forest Regressor, achieves robust performance, explaining approximately 59% of the variance in CLV.This repository contains the code, documentation, and analysis for the project, making it reproducible and adaptable for similar predictive modeling tasks.


DATASET DESCRIPTION

The analysis is based on the amazon_sales_dataset_2019_2024_corrected.xlsx dataset, which includes 5,000 entries and 15 columns capturing customer transactions on Amazon. Key columns relevant to the analysis include:

A. Customer ID: Unique identifier for each customer.

B. Order Date: Date of the purchase.

C. Product Name: Name of the purchased product.

D. Order Value: Monetary value of the order.


DATA QUALITY

A single missing value was identified in the Product Name column and imputed with the placeholder "Unknown". No other significant data quality issues (e.g., duplicates, inconsistencies) were observed after initial exploration.


METHODOLOGY

The project follows a structured pipeline to preprocess the data, engineer features, and build a predictive model for CLV.

3.1. DATA PREPROCESSING

A. Cleaning: Handled missing values and ensured consistent data types (e.g., converting dates to datetime format).

B. Exploratory Data Analysis (EDA): Conducted to understand distributions, correlations, and trends in customer purchasing behavior.

C. Data Splitting: Split the dataset into training (80%) and testing (20%) sets to evaluate model performance.

3.2. FEATURE ENGINEERING

To capture customer behavior, the following RFM-based features were engineered:

A. Customer Lifetime Value (CLV): Computed as the sum of all order values for each customer, representing their total monetary contribution.

B. Recency: Calculated as the number of days since a customer's most recent purchase, relative to the latest date in the dataset.

C. Frequency: Count of total orders placed by each customer.

D. Monetary: Total amount spent by each customer (equivalent to CLV in this context but used as an input feature for RFM analysis).

These features were derived using pandas and aggregated by Customer ID.

3.3. PREDICTIVE MODELING

A supervised machine learning approach was employed to predict CLV. The following regression models were evaluated:

A. Linear Regression: Baseline model assuming linear relationships.

B. Ridge Regression: Regularized linear model to mitigate overfitting.

C. Lasso Regression: Regularized model with feature selection capabilities.

D. ElasticNet Regression: Combines L1 and L2 regularization.

E. Decision Tree Regressor: Non-linear model capturing complex relationships.

F. Random Forest Regressor: Ensemble model combining multiple decision trees for improved accuracy and robustness.

A preprocessing pipeline using scikit-learn was implemented to:

A. Standardize numerical features (mean = 0, variance = 1).

B. Handle categorical variables (if applicable, though none were used in the final model).


3.4. HYPERPARAMETER OPTIMIZATION

The Random Forest Regressor was selected for its superior performance during initial model evaluation. Hyperparameter tuning was performed using GridSearchCV to optimize the following parameters:

A. n_estimators: Number of trees in the forest.

B. max_depth: Maximum depth of each tree.

C. min_samples_split: Minimum number of samples required to split a node.

D. min_samples_leaf: Minimum number of samples required at a leaf node.

The optimal hyperparameters were:

A. max_depth: None (unrestricted depth)

B. min_samples_leaf: 1

C. min_samples_split: 2

D. n_estimators: 100


MODEL EVALUATION

The tuned Random Forest Regressor was evaluated on the test set using the following metrics:

A. Mean Squared Error (MSE): Measures the average squared difference between predicted and actual CLV.

B. Root Mean Squared Error (RMSE): Square root of MSE, providing error in the same units as CLV.

C. Mean Absolute Error (MAE): Average absolute difference between predicted and actual CLV.

D. R-squared (R²): Proportion of variance in CLV explained by the model.


RESULTS

METRIC                   VALUE

MSE                   1,118,158.07

RMSE                  1,057.43

MAE                   737.99

R²                    0.5878


The R² value of 0.5878 indicates that the model explains approximately 59% of the variance in CLV, demonstrating moderate predictive power. The RMSE and MAE suggest that predictions are reasonably accurate but could benefit from further refinement.

CONCLUSION AND FUTURE WORK

This project successfully demonstrates the application of machine learning to predict Customer Lifetime Value using Amazon sales data. The tuned Random Forest Regressor provides a robust foundation for forecasting CLV, enabling businesses to identify high-value customers and tailor marketing strategies accordingly.


FUTURE WORK

A. Feature Expansion: Incorporate additional data sources, such as customer demographics, product categories, or external behavioral data (e.g., social media engagement).

B. Advanced Models: Explore ensemble methods like Gradient Boosting (e.g., XGBoost, LightGBM) or deep learning approaches to improve predictive accuracy.

C. Real-Time Deployment: Develop an API or production pipeline to deliver real-time CLV predictions for new customers.

D. Customer Segmentation: Apply clustering techniques (e.g., K-means, DBSCAN) to group customers based on RFM features for targeted marketing campaigns.

E. Feature Importance Analysis: Conduct a deeper analysis of feature contributions to understand key drivers of CLV.


INSTALLATION AND USAGE

Prerequisites

Python 3.8+

Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

Installation

Clone the repository:bash

git clone 

https://github.com/your-username/amazon-clv-prediction.git

cd amazon-clv-prediction

Install dependencies:

bash

pip install -r requirements.txt

Place the amazon_sales_dataset_2019_2024_corrected.xlsx file in the project directory or update the file path in the script.

UsageRun the main script to preprocess data, train the model, and evaluate results:bash

python clv_prediction.py

The script will:

A. Load and preprocess the dataset.

B. Engineer RFM features.

C. Train and tune the Random Forest Regressor.

D. Output model performance metrics and visualizations.

Contributions are welcome! To contribute:

A. Fork the repository.

B. Create a new branch (git checkout -b feature-branch).

C. Commit your changes (git commit -m "Add feature").

D. Push to the branch (git push origin feature-branch).

E. Open a pull request.

Please ensure your code follows PEP 8 guidelines and includes appropriate documentation.LicenseThis project is licensed under the MIT License. See the LICENSE file for details.
