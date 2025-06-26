# Task 3: Linear Regression

This repository contains my solution for Task 3 of the AI & ML Internship. The goal of this task is to implement simple and multiple linear regression using the Housing Price Prediction dataset and evaluate the model using various performance metrics.

---

## Objective

Implement and understand simple & multiple linear regression.

---

## Files Included

| File Name                | Description                            |
|-------------------------|----------------------------------------|
| `Housing.csv`           | Raw dataset used for this task         |
| `linear_regression.ipynb` | Jupyter Notebook with all implementation steps |
| `screenshots/`          | Output plots and visualizations        |
| `README.md`             | Project documentation                  |

---

## What I Did

1. **Data Preprocessing**
I started by loading the housing dataset and exploring its structure using df.info() and df.describe(). This helped me identify data types and spot missing or categorical values. I then handled the categorical features such as mainroad, guestroom, and furnishingstatus using one-hot encoding so they could be used effectively in the model.

2. **Splitting the Data**
Once the data was cleaned and encoded, I separated the target column (price) from the feature set. I then split the data into training and testing sets using an 80-20 ratio with train_test_split.

3. **Model Training**
I trained a linear regression model using sklearn.linear_model.LinearRegression. This allowed the model to learn relationships between the features and the house prices based on the training data.

4. **Model Evaluation**
After training, I evaluated the model’s performance on the test data using three key metrics:

- MAE (Mean Absolute Error) measures the average magnitude of errors in predictions.

- MSE (Mean Squared Error) penalizes larger errors more significantly.

- R² Score indicates how well the model explains the variance in the target variable.

5. **Visualization & Interpretation**
Following the training of the linear regression model, I analyzed the coefficients to see how each feature contributed to the prediction of house price. Here is my interpretation from the values:

The coefficient for area is around $236, implying that for every square foot bought, the predicted price of the house increases by $236, if all other feature values are held constant. This confirms that area has an implied positive and direct impact on price. 

Bedrooms and bathrooms also have a positive impact on the prediction, with one bathroom increasing the price by about $10 lakhs - the most of all numeric features. In this dataset, it appears that the number of bathrooms buyers are willing to pay for is not just placed moderately higher up the hierarchy than area, but are given a heavy weighting. An additional bathroom is required by many homeowners.

There are also multiple features where each additional increment in stories or parking space were positively weighted, contributing at least about $4–5 lakhs to the price predicting. Therefore, area was not the only feature for which some homeowners wanted more of, where additional stories or more parking space ranked higher on many buyers lists.

For the binary feature strikes in the regression coefficients, the presence of airconditioning_yes on the house increases the price by about $7.9 lakhs, closely followed by hotwaterheating_yes and prefarea_yes. Although the higher added cost on airconditioning might have been viewed from a comfort perspective, the higher costs of hot water heating (in possibly premium areas where heating and cooling come with little cost) emphasizes that buyers are prepared to pay more for comfort oriented features, and that like their hot water arangements, many home buyers are pre-disposed to pay for premium place preconditions buying homes on pre-determined premuim properties. 

Basement, guestroom, or main road access also were contributing positively to dollar values by about $2–4 lakhs for each of the features in this dataset too. These features may help to suggest that upper stratum buildings likely had higher construction standards or were in better accessible parts of town.

## Evaluation Metrics

| Metric | Value            |
|--------|------------------|
| MAE    | 970043.40        |
| MSE    | 1754318687330.66 |
| R²     | 0.65             |

These values suggest that the model has learned a reasonably good fit. The R² value of 0.65 means that approximately 65% of the variation in house prices is explained by the features used. While there's room for improvement, especially with advanced models or better feature engineering, this provides a solid foundation for linear regression.

---

## What I Learned
How to preprocess categorical and numerical data

The intuition behind simple and multiple linear regression

How to interpret coefficients and evaluate model performance

The importance of using metrics like MAE, MSE, and R² for regression tasks

## Tools & Libraries Used

- Python 3.12
- `pandas`, `numpy` — data handling
- `scikit-learn` — model training & evaluation
- `matplotlib` — visualization

---

## How to Run This Project

1. Clone the repository:
   ```bash
   git clone https://github.com/anmolthakur74/task-3-linear-regression.git
   cd task-3-linear-regression
   ```

2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

3. Open the notebook:
   ```
   jupyter notebook linear_regression.ipynb
   ```

---

## Author

**Anmol Thakur**

GitHub: [anmolthakur74](https://github.com/anmolthakur74/)

