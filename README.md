# SUV Market Value Predictor (ML Regression)

## Overview
This project focuses on developing a robust predictive model to estimate the market price (in USD) of SUV vehicles in Argentina. Using a real-world dataset scraped from Mercado Libre between May 13th and 30th, 2025, the model accounts for key variables such as brand, model, age, mileage, and color.

## Technical Objective
The goal was to build a machine learning pipeline capable of predicting vehicle prices for any combination of features present in the dataset. The model's performance was rigorously evaluated against a test set using the Root Mean Squared Error (RMSE) metric to ensure precision and reliability.

## Key Features & Business Insights
* **Predictive Modeling:** End-to-end ML pipeline from data preprocessing to model deployment.
* **Depreciation Analysis:** Investigated the specific impact of mileage and years of use on vehicle pricing.
* **Market Arbitrage Tool:** Developed logic to detect undervalued vehicles—identifying listings priced significantly below their expected market value to maximize potential investment gains.
* **Color Premium Ranking:** Analysis of how different vehicle colors affect relative resale prices.

## Tech Stack
* **Language:** Python
* **Libraries:** Scikit-learn (Regression), Pandas (Data Manipulation), NumPy, Matplotlib/Seaborn (Visualization).

## Documentation
The repository contains:
* `Notebook_SUVs.ipynb`: Full development process, EDA, and final model.
* `pf_suvs_i302_1s2025.csv`: The core dataset containing real market listings.
* Technical report and original project requirements (available in Spanish).

---
*Developed as part of the Artificial Intelligence Engineering program at Universidad de San Andrés (UdeSA).*
