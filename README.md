# 🚀 Boost Credit Scoring Service

![Credit Score](https://img.shields.io/badge/Credit-Scoring-blue) 
![API](https://img.shields.io/badge/Type-API-green)

A comprehensive credit scoring service that assesses financial risk using machine learning and traditional scoring models.

## 📋 Overview

The service evaluates customers based on eight core financial metrics:

| Metric | Description |
|--------|-------------|
| **Revenue Strength** | Measures income generation over time |
| **Debt-to-Equity Ratio** | Assesses financial leverage |
| **Payment History** | Tracks consistency in debt repayment |
| **Cash Reserves** | Reflects financial cushioning |
| **Business Longevity** | Evaluates stability and experience |
| **Industry Risk Score** | Contextualizes sector volatility |
| **Late Payments** | Identifies delinquency patterns |
| **Credit Utilization** | Examines borrowed capital usage |

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