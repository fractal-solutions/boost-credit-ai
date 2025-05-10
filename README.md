# 🚀 Boost Credit Scoring Service

![Credit Score](https://img.shields.io/badge/Credit-Scoring-blue) 
![API](https://img.shields.io/badge/Type-API-green)

A comprehensive credit scoring service that assesses financial risk using machine learning and traditional scoring models.

## 📊 Industry Risk Model Metrics & Weighting  

| Metric                | Description                                         | Weight | Ideal Weight | Category                   |
|-----------------------|----------------------------------------------------|--------|--------------|---------------------------|
| **Revenue Strength**  | Measures income generation over time              | 10%    | ~8-10%       | Business Fundamentals (30%) |
| **Debt-to-Equity Ratio** | Assesses financial leverage                    | 15%    | ~15%         | Business Fundamentals (30%) |
| **Payment History**   | Tracks consistency in debt repayment              | 35%    | ~35%         | Payment Behavior (40%)     |
| **Cash Reserves**     | Reflects financial cushioning                     | 5%     | ~4-5%        | Business Fundamentals (30%) |
| **Business Longevity** | Evaluates stability and experience               | 5%     | ~3-5%        | Risk Factors (30%)         |
| **Industry Risk Score** | Contextualizes sector volatility                 | 3%     | ~3-5%        | Risk Factors (30%)         |
| **Late Payments**     | Identifies delinquency patterns                   | 2%     | ~2%          | Payment Behavior (40%)     |
| **Credit Utilization** | Examines borrowed capital usage                  | 30%    | ~30%         | Payment Behavior (40%)     |
| **Credit History**    | Reviews past financial borrowing and repayment    | 7.5%   | ~5-10%       | Risk Factors (30%)         |

## 🔍 Overview
This table provides a **structured credit risk assessment framework**, ensuring **balanced weight distribution** across **payment behavior, business fundamentals, and risk factors**. The **ideal weight column** helps maintain industry-aligned scoring.

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