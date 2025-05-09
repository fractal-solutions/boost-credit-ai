import NeuralNetwork from './neural_network.js';
import { calculateFinalCreditScore } from './credit-scorer.js';

// Define interest rate relevant features and ranges
const INTEREST_RATE_FEATURES = {
    // Model scores (normalized to 0-1)
    MODEL_SCORES: {
        xgboost: [0, 1150],
        neuralNetwork: [300, 850],
        regression: [300, 850],
        ordinal: [300, 850]
    },
    
    // Key business metrics that affect interest rates
    BUSINESS_METRICS: {
        paymentHistory: [0, 1],
        creditUtilization: [0, 1],
        cashReserves: [0, 15],
        debtToEquity: [0, 3],
        industryRisk: [0, 10]  
    },
    
    // Interest rate ranges
    RATE_RANGES: {
        min: 0.00,
        max: 0.39,
        baseline: 0.10,
        industryRiskAdjustment: 0.02  // 2% per risk level above threshold
    }
};

export class InterestRateModel {
    constructor() {
        // 8 inputs: 4 model scores + 4 key business metrics
        this.nn = new NeuralNetwork(
            9,
            [16, 8],
            1,
            'sigmoid'
        );
        this.creditScore = 0;
    }

    normalizeInput(features) {
        const {MODEL_SCORES, BUSINESS_METRICS} = INTEREST_RATE_FEATURES;
        
        return [
            // Normalize model scores
            (features.xgboostScore - MODEL_SCORES.xgboost[0]) / (MODEL_SCORES.xgboost[1] - MODEL_SCORES.xgboost[0]),
            (features.nnScore - MODEL_SCORES.neuralNetwork[0]) / (MODEL_SCORES.neuralNetwork[1] - MODEL_SCORES.neuralNetwork[0]),
            (features.regressionScore - MODEL_SCORES.regression[0]) / (MODEL_SCORES.regression[1] - MODEL_SCORES.regression[0]),
            (features.ordinalScore - MODEL_SCORES.ordinal[0]) / (MODEL_SCORES.ordinal[1] - MODEL_SCORES.ordinal[0]),
            
            // Normalize business metrics
            features.paymentHistory,
            features.creditUtilization,
            features.cashReserves / BUSINESS_METRICS.cashReserves[1],
            features.debtToEquity / BUSINESS_METRICS.debtToEquity[1],
            features.industryRisk / BUSINESS_METRICS.industryRisk[1] 
        ];
    }

    calculateBaseRate(creditScore) {
        const {min, max, baseline} = INTEREST_RATE_FEATURES.RATE_RANGES;
        const maxCreditScore = 850;
        this.creditScore = creditScore;
        
        
        // Higher scores = lower rates
        return baseline + (1 - (creditScore / maxCreditScore)) * (max - min);
    }

    train(trainingData) {
        const normalizedData = trainingData.map(data => ({
            input: this.normalizeInput(data.features),
            output: [data.interestRate]
        }));

        this.nn.train(normalizedData, 0.001, 1000, true);
    }

    predict(features) {
        const normalizedInput = this.normalizeInput(features);
        const prediction = this.nn.forward(normalizedInput)[0];

        const finalCreditScore = calculateFinalCreditScore(features.nnScore, features.regressionScore, features.ordinalScore, features.ficoScore, features.xgboostScore)

        const baseRate = this.calculateBaseRate(finalCreditScore);
        

        // Calculate risk multiplier with reduced impact
        const riskMultiplier = this.calculateRiskMultiplier(features);
        
        // Adjust final rate calculation to be less extreme
        const riskAdjustment = (riskMultiplier - 1) * 0.25; // Reduce impact of risk multiplier
        const finalRate = baseRate * (1 + riskAdjustment);
        
        // Ensure rates stay within reasonable bounds
        const cappedFinalRate = Math.min(
            INTEREST_RATE_FEATURES.RATE_RANGES.max,
            Math.max(INTEREST_RATE_FEATURES.RATE_RANGES.min, finalRate)
        );

        // Normalize confidence to 0-100% range
        const normalizedConfidence = Math.min(100, prediction * 100);
        
        return {
            baseRate: (baseRate * 100).toFixed(2) + '%',
            adjustedRate: (cappedFinalRate * 100).toFixed(2) + '%',
            confidence: normalizedConfidence.toFixed(1) + '%',
            riskFactors: this.analyzeRiskFactors(features),
            riskMultiplier: riskMultiplier.toFixed(2),
            creditScore: this.creditScore
        };
    }


    calculateRiskMultiplier(features) {
        let multiplier = 1.0;
        
        // Reduced impact percentages
        // Payment history impact (0-15% adjustment)
        if (features.paymentHistory < 0.9) multiplier += 0.15;
        else if (features.paymentHistory < 0.95) multiplier += 0.075;
        
        // Credit utilization impact (0-12.5% adjustment)
        if (features.creditUtilization > 0.7) multiplier += 0.125;
        else if (features.creditUtilization > 0.5) multiplier += 0.075;
        
        // Cash reserves impact (0-10% adjustment)
        if (features.cashReserves < 0.5) multiplier += 0.10;
        else if (features.cashReserves < 1.0) multiplier += 0.05;
        
        // Debt-to-equity impact (0-12.5% adjustment)
        if (features.debtToEquity > 2.0) multiplier += 0.125;
        else if (features.debtToEquity > 1.5) multiplier += 0.075;

        if (features.industryRisk > 7) multiplier += 0.15;  // High risk
        else if (features.industryRisk > 5) multiplier += 0.075;  // Medium risk
        
        return multiplier;
    }

    analyzeRiskFactors(features) {
        const riskFactors = [];
        
        if (features.creditUtilization > 0.7) 
            riskFactors.push('High Credit Utilization');
        if (features.debtToEquity > 2) 
            riskFactors.push('High Debt-to-Equity Ratio');
        if (features.paymentHistory < 0.9) 
            riskFactors.push('Payment History Concerns');
        if (features.cashReserves < 0.5) 
            riskFactors.push('Low Cash Reserves');

        // industry risk factor
        if (features.industryRisk > 7)
            riskFactors.push('High Industry Risk');
        else if (features.industryRisk > 5)
            riskFactors.push('Moderate Industry Risk');
                
        
        return riskFactors;
    }
}