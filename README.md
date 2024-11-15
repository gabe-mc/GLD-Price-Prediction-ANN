# SPDR Gold Shares ETF (GLD) Prediction Abstact Neural Network

## Overview
This project aims to predict the price of the SPDR Gold Shares ETF ($GLD) using an Abstract Neural Network. Predictions are made using data from a collection of related stocks and commodities' closing prices from the end of the trading session, effectivly forcasing the next trading day's closing price for $GLD. 

This project combines findings from several research papers written on the modeling of deep-learning gold pricing models, which are [referenced below](#references).

## Table of Contents

1. [Dataset](#dataset)
2. [Model Architecture](#model-architecture)
3. [Training Process](#training-process)
4. [Results](#results)
5. [Installation](#installation)
6. [Usage](#usage)
7. [References](#references)
   
## Dataset
The training data for this model was largly chosen based off a coorelation analysis performed in Iftikhar ul Sami's *Predicting Future Gold Rates using Machine Learning Approach*, which ranked the stock price of Wheaton Precious Metals ($WPM) as the item with higest correlation in their sample. Their other highly ranked items included the spot price of silver, oil, and palladium, as well as the US Treasury's 4 Week bill rate. 

To account for the seasonality of gold prices, which experience a boost near Chinese New Year and other important cultural holidays that involve the purchasing of gold, month was included in the sample, an addition on ul Sami's chosen data.

| Feature         | Description                        | Data Source |
|-----------------|------------------------------------|-----------|
| `Wheaton Precious Metals`          | The stock price of $WPM, a large Canadian gold mining corporation. | [Financial Modelling Prep API](https://site.financialmodelingprep.com/)     |
| `Silver Price`        | Silver spot price at day-end. | [Financial Modelling Prep API](https://site.financialmodelingprep.com/)   |
| `Palladium Price`        | Palladium spot price at day-end   | [Financial Modelling Prep API](https://site.financialmodelingprep.com/)     |
| `Oil Price`       | Crude Oil spot price at day-end | [Financial Modelling Prep API](https://site.financialmodelingprep.com/)     |
| `4 Month US Treasury Bill`           | United States Federal Reserve's 4 week treasury bill rate | [Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/series/DTB4WK)     |
| `Month`          | The current month | Hand-Tagged     |
| `$GLD Price`     | Adjusted closing price of SPDR Gold Shares | [Financial Modelling Prep API](https://site.financialmodelingprep.com/)     |

Data was prepared using Python's Pandas module, to account for any time discrepancy between items that do not trade on public holidays and weekends, and those which do. 

In accordance with ul Sami's findings, this project only included the previous 250 trading days in the dataset. On top of that, a 75-15 data split was used between training and validation. 

## Model Architecture
The model powering is a three-layer feedforward neural network with:

1. **Input Layer**: Accepts 9 input features related to $GLD and market data.
2. **Hidden Layers**: 
   - Layer 1: 12 neurons, applies a linear transformation followed by ReLU activation.
   - Layer 2: 12 neurons, applies a linear transformation followed by ReLU activation.
3. **Output Layer**: Produces a single output representing the predicted $GLD price.

## Training Process
Input features are normalized between 0 and 1 for consistent training, and Torch's dataloader is used to batch training data. No shuffling is done.
The Mean Absolute Error (MAE) loss function (torch.L1Loss) calculates the error between predictions and actual values.
The Adam optimizer adjusts model weights to minimize the loss.
Training was run for 700,000 epochs with loss values recorded to monitor performance.

## Results
Using the 15% of the original dataset that was set aside for prediciton, this model is able to predict the price of GLD within a 2% margin of error.
## Installation
Instructions for setting up the environment and installing necessary libraries.

```bash
# Clone this repository
git clone <repo_url>

# Navigate into the project directory
cd project-directory

# Install dependencies
pip install -r requirements.txt
```
## Usage
For usage of the model, the Predict function in ```Predictor.py is`` able to predict the next GLD price based on the parameters fed to the function. If you connect your own API key for Financial Modelling Prep (in an enviroment variable called API_Key), the ```main.py``` file can be run, which will print the current day's GLD prediction. 

## References

1. **Modeling Gold Price via Artificial Neural Network**  
   *Journal of Economics, Business and Management*, Vol. 3, No. 7, July 2015  
   Authors: Hossein Mombeini and Abdolreza Yazdani-Chamzini  
   [Read the paper](https://www.joebm.com/papers/269-T20013.pdf)

2. **Predicting Future Gold Rates using Machine Learning Approach**  
   *International Journal of Advanced Computer Science and Applications*, January 2018  
   Author: Iftikhar ul Sami  
   [Read the paper](https://www.researchgate.net/publication/322222520_Predicting_Future_Gold_Rates_using_Machine_Learning_Approach)

3. **Predicting Stock Price based on the Residual Income Model, Regression with Time Series Error, and Quarterly Earnings**  
   February 2007  
   Author: Huong Ngo Higgins  
   [Read the paper](https://www.researchgate.net/publication/228536778_Predicting_Stock_Price_based_on_the_Residual_Income_Model_Regression_with_Time_Series_Error_and_Quarterly_Earnings)
