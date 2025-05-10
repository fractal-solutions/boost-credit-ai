# Boost Credit Scoring Service Specification

## Overview  
This document outlines the specifications for the credit scoring service that assesses financial risk based on key customer metrics. The service is implemented as an API that takes financial data as input and returns a credit score and related predictions.

---

## Core Components

### Financial Metrics  
The service examines eight core aspects of a customer's financial behavior:

#### 1. Revenue Strength  
- **Purpose**: Measures income generation over time  
- **Threshold**: ≥ KES 1.0M annual revenue  
- **Assessment**: Determines financial sustainability and growth capacity  
- **Impact**: Higher revenue strength indicates lower risk  

#### 2. Debt-to-Equity Ratio  
- **Purpose**: Assesses leverage  
- **Threshold**: ≤ 1.3 ratio  
- **Assessment**: Shows how much financing comes from debt vs. owner investment  
- **Impact**: Higher ratios may indicate higher financial risk  

#### 3. Payment History  
- **Purpose**: Tracks past consistency in debt repayment  
- **Threshold**: ≥ 78% on-time payments  
- **Assessment**: Signals reliability or risk  
- **Impact**: Consistent payment history reduces perceived risk  

#### 4. Cash Reserves  
- **Purpose**: Reflects financial cushioning  
- **Threshold**: ≥ KES 0.5M  
- **Assessment**: Indicates ability to withstand economic downturns  
- **Impact**: Larger reserves reduce risk of default  

#### 5. Business Longevity  
- **Purpose**: Evaluates stability and experience  
- **Threshold**: ≥ 2 years in business  
- **Assessment**: Older businesses often seen as lower risk  
- **Impact**: Longer history typically correlates with lower risk  

#### 6. Industry Risk Score  
- **Purpose**: Contextualizes sector-specific volatility  
- **Threshold**: ≤ 7 (scale 1-10)  
- **Assessment**: Adjusts expectations based on economic trends  
- **Impact**: Higher risk industries require higher scores  

#### 7. Late Payments  
- **Purpose**: Identifies delinquency patterns  
- **Threshold**: ≤ 6 late payments  
- **Assessment**: Crucial for predicting future default likelihood  
- **Impact**: More late payments indicate higher risk  

#### 8. Credit Utilization  
- **Purpose**: Examines borrowed capital usage  
- **Threshold**: ≤ 65% utilization  
- **Assessment**: Balances financial flexibility against excessive reliance on debt  
- **Impact**: High utilization may indicate financial stress  

---

### FICO SBSS Scoring Model  
The service is built on the FICO Small Business Scoring Service (SBSS) model, which is the primary scoring mechanism. All metrics feed directly into the SBSS calculation.

#### SBSS Implementation  
- Core scoring model with standardized score (ranging 0-300 extrapolated to scale from 300 through 850 to allow across the board rating)  
- Modified weights to better reflect our risk assessment priorities  
- All metrics are normalized and weighted according to SBSS standards  

## Industry-Standard Credit Scoring Weight Adjustments
The weight adjustments in credit scoring models are informed by **financial institutions, fintech trends, and regulatory frameworks**. These refinements ensure **accurate risk assessment** while optimizing lending decisions for SMEs.
The FICO SBSS model uses the following hierarchical weighting structure:

| Category                | Metric               | Weight Within Category | Total Weight | Ideal Weight |
|-------------------------|----------------------|------------------------|--------------|--------------|
| **Payment Behavior (40%)** | Payment History      | 60%                   | 35%        | ~35-40%  |
|                         | Late Payments        | 10%                   | 2%        | ~2-5%    |
|                         | Credit Utilization   | 30%                   | 30%       | ~25-30%  |
| **Business Fundamentals (30%)** | Annual Revenue      | 40%                   | 10%        | ~8-12%   |
|                         | Cash Reserves        | 20%                   | 5%       | ~4-6%    |
|                         | Debt-to-Equity Ratio | 40%                   | 15%       | ~12-18%  |
| **Risk Factors (30%)**  | Industry Risk       | 35%                   | 3%          | ~3-5%    |
|                         | Years in Business    | 45%                   | 5%        | ~3-6%    |
|                         | Credit History       | 20%                   | 7.5%         | ~5-10%   |



### Sources Informing Weight Adjustments

#### 1. **FICO SBSS Model**
- Payment history weight adjusted to **60%**, as it is the strongest predictor of default risk.
- Credit utilization weight adjusted to **30%**, ensuring balanced risk assessment.
- Late payments reduced to **10%**, as their impact is more significant when coupled with poor payment behavior.

#### 2. **Basel II & IFRS 9 Guidelines**
- **Debt-to-equity ratio** adjusted to **40%**, reflecting financial stability and leverage concerns.
- **Cash reserves** weight reduced to **20%**, prioritizing revenue generation over static liquidity.

#### 3. **Fintech Lending Trends**
- **Annual revenue** weight raised to **40%**, as fintech lenders rely on real-time earnings metrics.
- **Industry risk** weight adjusted to **35%**, ensuring sector-specific volatility is considered.
- **Years in business** weight adjusted to **45%**, linking business longevity to financial stability.

#### 4. **World Bank SME Credit Scoring Framework**
- Aligns risk models with global lending best practices.
- Introduces **alternative credit indicators** to refine scoring methodologies.

#### 5. **Kenya Credit Reference Bureau Standards**
- Ensures compliance with **local financial stability metrics**.
- Adapts risk models based on **macroeconomic and sector-specific conditions**.


---

## Technical Implementation

### API Endpoints  
- **POST /predict** - Accepts financial metrics as input and returns comprehensive credit score predictions  
  - Input format: JSON array of 8 feature values in order:  
    `[annualRevenue, debtToEquity, paymentHistory, cashReserves, yearsInBusiness, industryRisk, latePayments, creditUtilization]`  
  - Returns: JSON response with predictions from all models and final score  

- **GET /health** - Health check endpoint  

---

# Credit Scoring System Analysis

## 1. XGBoost Implementation

### Implementation  
- Binary classifier (good/bad credit) with probability outputs  
- Uses sigmoid scaling to convert probabilities to credit scores (300-850 range)  
- Feature importance analysis included  
- Threshold-based training data generation  

### Pros  
- Excellent at handling threshold cases  
- Provides clear probability outputs  
- Feature importance helps explain decisions  
- Fast prediction times  

### Cons  
- Binary classification loses granularity  
- Less effective for edge cases between categories  

---

## 2. Neural Network Models

### Model Architecture Overview  
The credit scoring system employs three specialized neural network models that work together in an ensemble approach:

#### 1. Generic Neural Network (Base Model)  
**Purpose**: Continuous credit score prediction  

**Implementation**:  
- Input layer: 8 nodes (one per financial feature)  
- Hidden layers: [16, 10] nodes with Swish activation  
- Output layer: 1 node (normalized credit score 0-1)  
- Training: 3000 epochs with learning rate 0.001  

**Functionality**:  
- Predicts raw credit scores (300-850 range)  
- Uses denormalization to convert output to FICO scale  
- Provides baseline prediction for ensemble  

**Pros**:  
- Simple architecture  
- Fast training and prediction  
- Good for initial screening  

**Cons**:  
- Lacks specialized handling of credit categories  
- May oversimplify complex patterns  

#### 2. Credit Regressor with Binning  
**Purpose**: Score prediction with built-in categorization  

**Implementation**:  
- Input layer: 8 nodes  
- Hidden layers: [32, 16, 16, 8] with Swish activation  
- Output layer: 1 node (normalized score)  
- Post-processing: Maps to 7 credit bins (300-850)  

**Functionality**:  
- Predicts continuous scores like base model  
- Automatically categorizes into credit ranges:  
  - Very Poor (300-579)  
  - Poor (580-669)  
  - Fair (670-739)  
  - Good (740-799)  
  - Excellent (800-850)  

**Pros**:  
- Built-in categorization  
- Deeper architecture captures more patterns  
- Clear output interpretation  

**Cons**:  
- Hard bin boundaries can create discontinuities  
- More computationally intensive  

#### 3. Ordinal Credit Classifier  
**Purpose**: Specialized ordinal classification  

**Implementation**:  
- Input layer: 8 nodes  
- Hidden layers: [32, 16, 8] with ReLU activation  
- Output layer: 7 nodes (one per category) with softmax  

**Functionality**:  
- Directly predicts probability distribution across 7 ordered categories  
- Uses custom softmax for confidence scoring  
- Provides weighted score within predicted range  

**Pros**:  
- Optimized for ordinal data (credit tiers)  
- Provides confidence scores  
- Handles category boundaries better  

**Cons**:  
- Most complex model  
- Requires careful calibration  

---

### Ensemble Approach

#### Integration Methodology  
1. **Base Prediction**: All models generate independent scores  
2. **Discrepancy Analysis**: System checks for model agreement  
3. **Dynamic Weighting**:  
   - Models closer to FICO SBSS baseline get higher weights  
   - Confidence scores influence final weighting  
4. **Final Score Calculation**:  
   - Weighted average of agreeing models  
   - Special handling when XGBoost disagrees  
   - Capping to prevent extreme values  

#### Key Features  
- **Self-Correcting**: Automatically reduces weight of outliers  
- **Explainable**: Tracks which models contributed most  
- **Adaptive**: Handles edge cases with custom logic  

#### Advantages  
- Combines strengths of different approaches  
- More robust than any single model  
- Provides built-in validation through consensus  

#### Limitations  
- Increased computational cost  
- More complex debugging  
- Requires careful calibration of weights  

#### Pros  
- Handles non-linear relationships well  
- Ordinal classifier matches credit scoring needs  
- Provides confidence scores  
- Can capture complex patterns  

#### Cons  
- Requires more training data  
- Harder to interpret than XGBoost  
- Training can be computationally expensive  

---

## 3. FICO SBSS Standardized Score

### Implementation  
- Traditional scoring model as baseline  
- Modified weights to reflect priorities  
- Comprehensive metric weighting system  

### Pros  
- Industry standard reference  
- Transparent calculation  
- Easy to explain to stakeholders  
- Proven effectiveness  

### Cons  
- Less flexible than ML approaches  
- Static weighting may not adapt to changes  
- Requires manual threshold tuning  

---

## 4. Interest Rate Model

### Implementation  
- Neural network with 9 inputs  
- Considers model scores and business metrics  
- Risk factor analysis  
- Dynamic rate adjustment  

### Pros  
- Comprehensive risk assessment  
- Clear risk factor identification  
- Rate adjustments based on multiple factors  
- Confidence scoring  

### Cons  
- Complex input requirements  
- Rate calculations may need calibration  
- Risk multipliers could be more granular  

---

## 5. Ensemble Methodology

### Implementation  
- Dynamic weighting system  
- Considers model agreement  
- FICO score as anchor  
- Special handling for edge cases  

### Pros  
- Balances strengths of different models  
- Reduces individual model biases  
- More stable predictions  
- Handles disagreements intelligently  

### Cons  
- Complex weighting logic  
- Potential over-reliance on FICO baseline  
- Final capping may hide true risk  

---

## 6. System Assessment

### Strengths  
- Comprehensive multi-model approach  
- Good balance of traditional and ML methods  
- Clear risk factor identification  
- API implementation for easy integration  
- Continuous learning planned  

### Weaknesses  
- Complex system with many components  
- Some redundancy between models  
- Documentation could be improved  
- Testing scenarios limited  

### Recommended Improvements  
1. Add more diverse training data  
2. Implement the planned continuous learning  
3. Simplify some overlapping model functionality  
4. Enhance documentation of decision logic  
5. Add more test cases for edge scenarios  
6. Consider model performance monitoring  

---

## Operational Details

### Training Data Enhancement  
The system will implement a continuous learning mechanism where:

1. **Completed Loan Tracking**:  
   - Loans that complete their full term will be added to the training dataset  
   - Actual performance (default/repaid) will be used as ground truth  

2. **Data Collection Process**:  
   - Loan applications that are approved and funded will be tracked  
   - Periodic updates will capture payment performance  
   - Final outcomes will be recorded upon loan completion  

3. **Model Retraining**:  
   - Scheduled retraining cycles will incorporate new data  
   - Performance metrics will be monitored for drift detection  
   - Model versions will be maintained for rollback capability  

(Implementation of this enhancement is planned for future release)  

### Example API Request  
```json
POST /predict
{
  "features": [5.0, 0.5, 0.94, 0.8, 4, 4, 1, 0.4]
}
```

### Example API Response  
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