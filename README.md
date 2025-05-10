# Boost Credit Scoring Service

This repository contains the implementation of a credit scoring service that assesses financial risk based on key customer metrics. The service provides API endpoints to predict credit scores using multiple machine learning models and traditional scoring approaches.

## Overview

The credit scoring service evaluates customers based on eight core financial metrics:
- Revenue Strength
- Debt-to-Equity Ratio  
- Payment History
- Cash Reserves
- Business Longevity
- Industry Risk Score
- Late Payments
- Credit Utilization

For detailed specifications including:
- Technical implementation details
- API endpoints
- Model architectures
- Scoring methodology

Please see the [Specification Document](./specification.md).

## Getting Started

To use this service:
1. Clone the repository
2. Install dependencies
3. Run the API server
4. Make POST requests to `/predict` endpoint with financial data

## Models Included

- XGBoost classifier
- Neural network ensemble
- FICO SBSS standardized score
- Interest rate model

See the [Specification Document](./specification.md) for complete details on model implementations and scoring methodology.