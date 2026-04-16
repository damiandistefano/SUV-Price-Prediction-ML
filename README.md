# SUV Market Value Predictor (ML Regression)

## Overview
[cite_start]This project focuses on developing a robust predictive model to estimate the market price (in USD) of SUV vehicles in Argentina[cite: 3]. [cite_start]Using a real-world dataset scraped from **Mercado Libre** between May 13th and 30th, 2025, the model accounts for key variables such as brand, model, age, mileage, and color[cite: 6].

## Technical Objective
[cite_start]The goal was to build a machine learning pipeline capable of predicting vehicle prices for any combination of features present in the dataset[cite: 8]. [cite_start]The model's performance was rigorously evaluated against a test set using the **Root Mean Squared Error (RMSE)** metric to ensure precision and reliability[cite: 10].

## Key Features & Business Insights
* [cite_start]**Predictive Modeling:** End-to-end ML pipeline from data preprocessing to model deployment[cite: 9].
* [cite_start]**Depreciation Analysis:** Investigated the specific impact of mileage and years of use on vehicle pricing[cite: 12].
* [cite_start]**Market Arbitrage Tool:** Developed logic to detect **undervalued vehicles**—identifying listings priced significantly below their expected market value to maximize potential investment gains[cite: 16, 18].
* [cite_start]**Color Premium Ranking:** Analysis of how different vehicle colors affect relative resale prices[cite: 13, 14].

## Tech Stack
* **Language:** Python
* **Libraries:** Scikit-learn (Regression), Pandas (Data Manipulation), NumPy, Matplotlib/Seaborn (Visualization).

## Documentation
The repository contains:
* [cite_start]`Notebook_SUVs.ipynb`: Full development process, EDA, and final model[cite: 4].
* [cite_start]`pf_suvs_i302_1s2025.csv`: The core dataset containing real market listings[cite: 6].
* [cite_start]Technical report and original project requirements (available in Spanish)[cite: 3, 9].

---
[cite_start]*Developed as part of the Artificial Intelligence Engineering program at Universidad de San Andrés (UdeSA)[cite: 1, 3].*