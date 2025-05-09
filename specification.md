# Boost Credit Scoring Service Specification

## Overview
This document outlines the specifications for the credit scoring service that assesses financial risk based on key customer metrics. The service is implemented as an API that takes financial data as input and returns a credit score and related predictions.

## Core Financial Metrics
The service examines eight core aspects of a customer's financial behavior with the following thresholds:

### 1. Revenue Strength
- **Purpose**: Measures income generation over time
- **Threshold**: ≥ KES 1.0M annual revenue
- **Assessment**: Determines financial sustainability and growth capacity
- **Impact**: Higher revenue strength indicates lower risk

### 2. Debt-to-Equity Ratio
- **Purpose**: Assesses leverage
- **Threshold**: ≤ 1.3 ratio
- **Assessment**: Shows how much financing comes from debt vs. owner investment
- **Impact**: Higher ratios may indicate higher financial risk

### 3. Payment History
- **Purpose**: Tracks past consistency in debt repayment
- **Threshold**: ≥ 78% on-time payments
- **Assessment**: Signals reliability or risk
- **Impact**: Consistent payment history reduces perceived risk

### 4. Cash Reserves
- **Purpose**: Reflects financial cushioning
- **Threshold**: ≥ KES 0.5M
- **Assessment**: Indicates ability to withstand economic downturns
- **Impact**: Larger reserves reduce risk of default

### 5. Business Longevity
- **Purpose**: Evaluates stability and experience
- **Threshold**: ≥ 2 years in business
- **Assessment**: Older businesses often seen as lower risk
- **Impact**: Longer history typically correlates with lower risk

### 6. Industry Risk Score
- **Purpose**: Contextualizes sector-specific volatility
- **Threshold**: ≤ 7 (scale 1-10)
- **Assessment**: Adjusts expectations based on economic trends
- **Impact**: Higher risk industries require higher scores

### 7. Late Payments
- **Purpose**: Identifies delinquency patterns
- **Threshold**: ≤ 6 late payments
- **Assessment**: Crucial for predicting future default likelihood
- **Impact**: More late payments indicate higher risk

### 8. Credit Utilization
- **Purpose**: Examines borrowed capital usage
- **Threshold**: ≤ 65% utilization
- **Assessment**: Balances financial flexibility against excessive reliance on debt
- **Impact**: High utilization may indicate financial stress

## FICO SBSS Scoring Model
The service is built on the FICO Small Business Scoring Service (SBSS) model, which is the primary scoring mechanism. All metrics feed directly into the SBSS calculation.

### SBSS Implementation
- Core scoring model with standardized score (ranging 0-300 extrapolated to scale from 300 through 850 to allow across the board rating)
- Modified weights to better reflect our risk assessment priorities
- All metrics are normalized and weighted according to SBSS standards

### Metric Weights
The FICO SBSS model uses the following hierarchical weighting structure:

| Category            | Metric               | Weight Within Category | Total Weight |
|---------------------|----------------------|------------------------|--------------|
| **Payment Behavior (35%)** | Payment History      | 50%                   | 17.5%        |
|                     | Late Payments        | 15%                   | 5.25%        |
|                     | Credit Utilization   | 35%                   | 12.25%       |
| **Business Fundamentals (35%)** | Annual Revenue      | 30%                   | 10.5%        |
|                     | Cash Reserves        | 35%                   | 12.25%       |
|                     | Debt-to-Equity Ratio | 35%                   | 12.25%       |
| **Risk Factors (30%)** | Industry Risk       | 40%                   | 12%          |
|                     | Years in Business    | 35%                   | 10.5%        |
|                     | Credit History       | 25%                   | 7.5%         |

## Implementation Status
The service is fully implemented with the following components:

### API Endpoints
- **POST /predict** - Accepts financial metrics as input and returns comprehensive credit score predictions
  - Input format: JSON array of 8 feature values in order:
    [annualRevenue, debtToEquity, paymentHistory, cashReserves, 
     yearsInBusiness, industryRisk, latePayments, creditUtilization]
  - Returns: JSON response with predictions from all models and final score

- **GET /health** - Health check endpoint

### Technical Components
- **Credit Scoring Models**:
  - XGBoost classifier with probability outputs
  - Neural Network regressor
  - Neural Network ordinal classifier
  - Credit score regression model
- **Supporting Models**:
  - Interest rate prediction model
- **Core Algorithms**:
  - Feature normalization
  - Threshold analysis
  - Score calculation and standardization

### Models Used
1. **XGBoost Classifier**:
   - Binary classification (good/bad credit)
   - Feature importance analysis
   - Probability outputs converted to credit scores

2. **Neural Networks**:
   - Generic neural network for score prediction
   - Regression model with binning
   - Ordinal classification model

3. **Interest Rate Model**:
   - Predicts interest rates based on credit scores
   - Identifies risk factors
   - Provides confidence scores

## Example API Request
```json
POST /predict
{
  "features": [5.0, 0.5, 0.94, 0.8, 4, 4, 1, 0.4]
}
```

## Example API Response
```json
{
  "features": {
    "annualRevenue": 5.0,
    "debtToEquity": 0.5,
    "paymentHistory": 0.94,
    "cashReserves": 0.8,
    "yearsInBusiness": 4,
    "industryRisk": 4,
    "latePayments": 1,
    "creditUtilization": 0.4
  },
  "predictions": {
    "xgboost": {
      "probability": 0.95,
      "score": 820,
      "category": "GOOD"
    },
    "neuralNetwork": {
      "score": 815,
      "category": "GOOD"
    },
    "regression": {
      "score": 825,
      "category": "GOOD"
    },
    "ordinalClassification": {
      "score": 830,
      "category": "GOOD"
    },
    "interestRate": {
      "baseRate": 5.2,
      "finalRate": 4.8,
      "confidence": 0.92
    },
    "finalScore": 825
  },
  "thresholdsMet": 8,
  "standardizedScore": 830
}