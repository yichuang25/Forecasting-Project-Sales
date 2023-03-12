# Forecasting-Project-Sales

## Introduction

- The Online Retail II data set gathered actual online retail data in two years from the UCI machine learning repository. The data set source is Dr. Daqing Chen, Course Director: MSc Data Science. chend **'@'** lsbu.ac.uk, School of Engineering, London South Bank University, London SE1 0AA, UK.
- The Online Retail data set contains all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011. The company mainly sells unique all-occasion giftware. Many customers of the company are wholesalers.
- Total sales will be the target variable and predicting variable will be time series.
- Time series analysis model (MA, AR, ARIMA, etc.) and machine learning model will be implemented to analyze and forecast future sales.

The purpose of this document is to provide a detailed technical overview of:

1. Data collecting and cleaning 
2. Data inclusions and exclusions
3. Predictive variable creation process 
4. Target variable definition
5. Iterative model building process
6. Metric evaluation process

## Data Processing

### Data Overview

We are doing Forecasting on UCI Online Retail Data Set. This dataset contains all the transaction information occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

### Data Extraction Process

The data is downloaded from UCI Machine Learning Repository, Online Retail Data.

### Data Diagnostics

A series of quality checks are performed on the data extract provide. These checks included:

- Attribute information
- Number of records
- Duplicate records if any
- Missing values in relevant fields

Attribution information:

1. InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
2. StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
3. Description: Product (item) name. Nominal.
4. Quantity: The quantities of each product (item) per transaction. Numeric.
5. InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction.
6. UnitPrice: Unit price. Numeric, Product price per unit in sterling.
7. CustomerID: Customer number. Nominal, a 5-digit integral unique number.
8. Country: Country name. Nominal, the name of the country where each customer resides.

The number of records: 

1. Year 2009-2010: 525461
2. Year 2010-2011: 541910

Duplicate records if any: There are no duplicate records.

### Modeling Data Creation

For each of the customers, the modeling data are collected and processed using the following steps. The below also addresses the feature engineering section. 

- Holidays: create features using time lags from previous 3 days to future 2 days [-3, 2]
- Time related features: month, year, week are created using datetime

## Target Variables

Total_Amount: quantity times price for each product which is the sales of the day

## Predictive Variables

A predictive variable is a variable used in algorithmic solutions to predict the target variable. During our analysis, we categorized predictive variables into two categories:

1. Direct variable – These variables were directly from the dataset that direct customers provided
2. Derived variable – These variables were created by manipulating the direct variables

### Variable List

1. Weather: Temperature, Seasons data.
2. Inventory: customer’s inventory. 
3. Holiday: US national holidays. Holiday has an effect on people’s purchasing patterns, for example, Black Friday, Christmas, etc. It is partly related to promotion. 
4. Features of time: Year, Month, Week. Elements of time are critical in the modeling because they capture seasonality in shipments. For example, the load of ice cream increases during summer times. 
5. Quantity: How many weeks was the product sold

## Pre-Modeling

- Before modeling, some analysis techniques were used to discover the seasonality and trend of the original dataset.
- Seasonal decomposition for the hourly and daily sales datasets can help us pick the dataset for modeling.
- The pattern and the cut-offs in ACF and PACF can help choose parameters for models such as ARIMA and SARIMAX.

### Train-Test Split

We split the dataset into training, validation, and testing. The ratio of each part is 8:1:1. 80% of the data was used to train the model, and about 10% data was used for validation during the hyperparameter tuning and model selection process. Another 10% of the data was used as Testing data to calculate the error of the best model. The Testing data will be split into three parts and evaluated each part of the data.

## Modeling

### Introduction

After the candidate predictive variables were finalized in the pre-modeling step, the team performed the necessary data mining and statistical iterations to develop and build the Algorithmic Solutions.

1. SARIMAX

	- Seasonal Auto-regressive Integrated Moving Average with eXogenous factors, is an extension of the ARIMA class of models.

	- SARIMA has the capability of dealing with seasonality and handling exogenous variables.

2. GLM
	- Because GLM cannot handle date-time type data, quarter, month, day in the month, and weekday are extracted from the date. Also, the holiday days in the UK From 2009 to 2011 are marked for tracking the holiday season trend. Daily weather data collected by the weather station at Heathrow Airport are also implemented.
	- Among OLS, Poisson, Gaussian, and Gamma regression in the GLM model, the Poisson regression yields the best result with MAPE = 0.324.

3. Prophet
	- Prophet is a procedure for forecasting time series based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series with substantial seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend and typically handles outliers well.

4. Neural Prophet
	- Neural Prophet is an easy-to-learn framework for interpretable time series forecasting. Neural Prophet is built on PyTorch and combines Neural Network and traditional time-series algorithms inspired by Facebook Prophet and AR-Net.

### Hyperparameters Selection of Algorithms

- SARIMAX

- - AR parameters
	- Differences
	- MA parameters
	- Seasonal component of the model for the AR parameters
	- Seasonal component of the model for the Differences
	- Seasonal component of the model for the MA parameters

- GLM

- - Different parameters combinations choosing
	- Link Function

- Prophet

- - change_prior_scale
	- change_point_range
	- holidays_prior_scale
	- interval_width
	- seasonality_prior_scale
	- uncertainty_samples

- Neural Prophet

- - batch_size
	- changepoints_range
	- daily_seasonality
	- d_hidden
	- epochs
	- growth
	- impute_missing
	- learning_rate
	- loss_func
	- normalize
	- num_hidden_layers
	- n_change_points
	- n_forecasts
	- optimizer
	- seasonality_mode
	- weekly_seasonality
	- yearly_seasonality

- ## File Description
  - Folder Explore
    - Data_Explore.ipynb: clean the data and generate Data/total_sales_09-10.pkl, Data/total_sales_10-11.pkl, Data/total_sales_09-10_cleaned.pkl and Data/total_sales_10-11_cleaned.pkl.
  - Folder Analysis&Modeling
    - Analysis.ipynb/Analysis_continued.ipynb: Visualizations and analysis on Data/total_sales_09-10.pkl, Data/total_sales_10-11.pkl.
    - data_filtered: aggregated amount by date and generate Data/sales_day.pkl and Data/sales_day_filtered.pkl(drop data outside of 95%).
    - GLM_originalData.ipynb/GLM_filteredDatai.ipynb: GLM model generated.
    - Holt-Winters.ipynb: Holt-Winters model generated.
    - prophet_originalData.ipynb/prophet_originalData: prophet model generated.
    - SARIMAX_originalData/SARIMAX_filteredData.ipynb: SARIMAX model generated.
    - neuralprophet_originalData.ipynb/neuralprophet_filteredData.ipynb: Neural Prophet model generated.
  - Folder Data
    - online_retail_II.xlsx: Raw data
    - sales_day.pkl: Sales amount aggregated by date
      - sales_day_filtered.pkl: Data outside of 95% range is dropped.
    - sales_half-day.pkl: Sales amount aggregated by half of the day
    - Sales09-10.pkl: Data from 2009 to 2010 after cleaning
      - Sales09-10_cleaned.pkl: weekday and hour are extracted
    - Sales10-11.pkl: Data from 2010 to 2011 after cleaning
      - Sales10-11_cleaned.pkl: weekday and hour are extracted
- ## Team Infomation

- Haolong Liu - hl3614@columbia.edu

- Yichen Huang - yichen.huang@columbia.edu

- Taichen Zhou - tz2555@columbia.edu
