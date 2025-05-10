# 🚀 Boost Credit Scoring Service

![Credit Score](https://img.shields.io/badge/Credit-Scoring-blue) 
![API](https://img.shields.io/badge/Type-API-green)

A comprehensive credit scoring service that assesses financial risk using machine learning and traditional scoring models.

## 📊 Industry Risk Model Metrics & Weighting  

| Metric                | Description                                         | Category                   | Weight | Ideal Weight |
|-----------------------|----------------------------------------------------|---------------------------|--------|--------------|
| **Revenue Strength**  | Measures income generation over time              | Business Fundamentals (30%) | 10%    | ~8-10%       |
| **Debt-to-Equity Ratio** | Assesses financial leverage                    | Business Fundamentals (30%) | 15%    | ~15%         |
| **Payment History**   | Tracks consistency in debt repayment              | Payment Behavior (40%)     | 35%    | ~35%         |
| **Cash Reserves**     | Reflects financial cushioning                     | Business Fundamentals (30%) | 5%     | ~4-5%        |
| **Business Longevity** | Evaluates stability and experience               | Risk Factors (30%)         | 5%     | ~3-5%        |
| **Industry Risk Score** | Contextualizes sector volatility                 | Risk Factors (30%)         | 3%     | ~3-5%        |
| **Late Payments**     | Identifies delinquency patterns                   | Payment Behavior (40%)     | 2%     | ~2%          |
| **Credit Utilization** | Examines borrowed capital usage                  | Payment Behavior (40%)     | 30%    | ~30%         |
| **Credit History**    | Reviews past financial borrowing and repayment    | Risk Factors (30%)         | 7.5%   | ~5-10%       |

## 🔍 Overview
This table consolidates **key SME lending risk factors**, providing a structured assessment framework with **category-based weight distribution**. The **ideal weight column** aligns with industry standards, ensuring accuracy in credit risk modeling.


## 📚 Documentation

For complete technical details including:
- API specifications
- Model architectures 
- Scoring methodology
- Implementation details

👉 See the [Full Specification Document](./specification.md)

---

## 🛠 Getting Started

### Prerequisites
- Node.js 16+ or Bun 1.2.6+


### Running the Service
```bash
npm run credit-scorer
```
or
```bash
bun credit-scorer
```
---

## 🔍 Models Overview

### 🎯 XGBoost Classifier
- Binary classification (good/bad credit)
- Probability outputs converted to scores

### 🧠 Neural Network Ensemble
- Multiple specialized architectures
- Continuous score prediction
- Built-in categorization

### 📊 FICO SBSS Standardized Score
- Industry standard reference
- Transparent calculation

### 💰 Interest Rate Model
- Dynamic rate adjustment
- Comprehensive risk assessment

---

## 📝 License
MIT © 2025 Boost Credit AI