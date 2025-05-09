const { XGBoost } = require('./xgboost.js');
import NeuralNetwork from './neural_network.js';
import { CreditRegressor, OrdinalCreditClassifier } from './credit_score_models.js';
import { InterestRateModel } from './interest_rate_model.js';

// Feature definitions and thresholds for good credit (Quick Classification Observation)
const THRESHOLDS = {
    MIN_ANNUAL_REVENUE: 1.0,       // Millions
    MAX_DEBT_TO_EQUITY: 1.3,       // Ratio
    MIN_PAYMENT_HISTORY: 0.78,     // 78% on-time payments
    MIN_CASH_RESERVES: 0.5,        // Millions
    MIN_YEARS_IN_BUSINESS: 2,      // Years
    MAX_INDUSTRY_RISK: 7,          // Scale 1-10
    MAX_LATE_PAYMENTS: 6,          // Count
    MAX_CREDIT_UTILIZATION: 0.65   // 65% maximum
};

//FICO SBSS Model weights
const MODIFIED_SBSS_WEIGHTS = {
    PAYMENT_BEHAVIOR: {
        weight: 0.40,
        metrics: {
            paymentHistory: 0.60,
            latePayments: 0.10,
            creditUtilization: 0.30
        }
    },
    BUSINESS_FUNDAMENTALS: {
        weight: 0.30,
        metrics: {
            annualRevenue: 0.40,
            cashReserves: 0.25,
            debtToEquity: 0.40
        }
    },
    RISK_FACTORS: {
        weight: 0.30,
        metrics: {
            industryRisk: 0.35,
            yearsInBusiness: 0.45,
            creditHistory: 0.20
        }
    }
};

// Define normalization ranges at the top level
export const FEATURE_RANGES = [
    [0, 50],    // Annual Revenue
    [0, 3],     // Debt to Equity
    [0, 1],     // Payment History
    [0, 10],     // Cash Reserves
    [0, 25],    // Years in Business
    [0, 10],    // Industry Risk
    [0, 20],    // Late Payments
    [0, 1]      // Credit Utilization
];

const X_train = [
    // Threshold edge cases - testing boundary conditions
    [
        1.0,   0.8,   0.78,  0.5,   2,   7,   6,   0.65,  // At threshold (borderline good)
        1.1,   1.3,   0.79,  0.51,  3,   6,   5,   0.64,  // Just above/below thresholds (good)
        0.9,   1.4,   0.77,  0.49,  1,   8,   7,   0.66   // Just missing thresholds (bad)
    ],

    // Strong performers in different categories
    [
        // High revenue, excellent metrics
        50.0,  0.5,   0.95,  5.0,   10,  3,   1,   0.30,  // Large stable business
        30.0,  0.7,   0.93,  3.0,   8,   4,   2,   0.40,  // Mid-size growing company
        20.0,  0.6,   0.94,  2.5,   7,   3,   1,   0.35,  // Successful medium business
        
        // Low revenue, excellent management
        1.5,   0.4,   0.98,  0.6,   3,   4,   0,   0.25,  // Small but excellent
        1.2,   0.5,   0.96,  0.55,  2,   3,   1,   0.30,  // Young but well-managed
        1.8,   0.3,   0.97,  0.7,   4,   5,   0,   0.35   // Small but strong metrics
    ],

    // Mixed performance cases
    [
        // Good revenue, poor management
        15.0,  1.8,   0.75,  0.3,   5,   8,   7,   0.80,  // Growth but poor controls
        12.0,  1.6,   0.77,  0.4,   4,   7,   8,   0.75,  // Size but poor metrics
        
        // Low revenue, good management
        1.3,   0.6,   0.92,  0.6,   3,   4,   2,   0.45,  // Small but well-managed
        1.6,   0.7,   0.90,  0.55,  2,   5,   3,   0.50   // Young but promising
    ],

    // Industry risk variations
    [
        // High risk, good management
        8.0,   0.8,   0.93,  1.2,   6,   8,   2,   0.45,  // Managing high risk well
        6.0,   0.9,   0.91,  1.0,   5,   9,   3,   0.50,  // Handling risk effectively
        
        // Low risk, poor management
        7.0,   1.7,   0.76,  0.3,   4,   3,   7,   0.85,  // Safe industry, poor execution
        5.0,   1.8,   0.75,  0.2,   3,   4,   8,   0.90   // Low risk but mismanaged
    ],

    // Payment history focus
    [
        // Strong payment history
        4.0,   0.9,   0.95,  0.8,   5,   6,   1,   0.55,  // Excellent payment record
        3.0,   1.0,   0.94,  0.7,   4,   5,   2,   0.60,  // Strong payment performance
        
        // Poor payment history
        6.0,   1.4,   0.70,  0.9,   6,   7,   9,   0.70,  // Payment issues
        5.0,   1.5,   0.72,  0.8,   5,   6,   8,   0.75   // Consistent late payments
    ],

    // Cash reserves variations
    [
        // Strong reserves
        10.0,  1.1,   0.88,  2.5,   7,   5,   3,   0.60,  // High cash reserves
        8.0,   1.2,   0.87,  2.0,   6,   6,   4,   0.62,  // Good cash position
        
        // Poor reserves
        12.0,  1.3,   0.86,  0.2,   8,   7,   5,   0.68,  // Cash flow issues
        9.0,   1.4,   0.85,  0.1,   7,   8,   6,   0.70   // Liquidity problems
    ]
];

import { promises as fs } from 'fs';
import path from 'path';

// Flatten the nested arrays into a single training set
const flattenedX_train = X_train.reduce((acc, curr) => acc.concat(curr), []);

// Group the data into rows of 8 features each
let processedX_train;
try {
    // Check if data directory exists, create if not
    await fs.mkdir(path.join(process.cwd(), 'data'), { recursive: true });
    
    // Try to load training data
    const data = await fs.readFile(path.join(process.cwd(), 'data', 'trainingset.json'), 'utf8');
    processedX_train = JSON.parse(data);
    console.log('Loaded training data from file');
} catch (err) {
    // File doesn't exist or error reading - create new data
    processedX_train = [];
    for (let i = 0; i < flattenedX_train.length; i += 8) {
        processedX_train.push(flattenedX_train.slice(i, i + 8));
    }
    
    // Save to file for next time
    await fs.writeFile(
        path.join(process.cwd(), 'data', 'trainingset.json'),
        JSON.stringify(processedX_train, null, 2)
    );
    console.log('Saved new training data to file');
}

// Calculate binary credit scores (1 for good, 0 for bad) based on all factors
const y_train = processedX_train.map(sample => {
    const meetsThresholds = [
        sample[0] >= THRESHOLDS.MIN_ANNUAL_REVENUE,
        sample[1] <= THRESHOLDS.MAX_DEBT_TO_EQUITY,
        sample[2] >= THRESHOLDS.MIN_PAYMENT_HISTORY,
        sample[3] >= THRESHOLDS.MIN_CASH_RESERVES,
        sample[4] >= THRESHOLDS.MIN_YEARS_IN_BUSINESS,
        sample[5] <= THRESHOLDS.MAX_INDUSTRY_RISK,
        sample[6] <= THRESHOLDS.MAX_LATE_PAYMENTS,
        sample[7] <= THRESHOLDS.MAX_CREDIT_UTILIZATION
    ];
    
    // Business needs to meet at least x out of 8 criteria for good credit
    const goodCreditScore = meetsThresholds.filter(Boolean).length >= 7;
    const score = meetsThresholds.filter(Boolean).length / 8;
    //return score;
    return goodCreditScore ? 1 : 0;
});

// Adjust model parameters
const model = new XGBoost({
    learningRate: 0.05,    // Reduced learning rate
    maxDepth: 6,          // Reduced depth
    minChildWeight: 2,    // Reduced min child weight
    numRounds: 100,       // Reduced rounds
    objective: 'binary'
});

// Train the model with binary labels
model.fit(processedX_train, y_train);

// Test data with scenario descriptions
const X_test = [
    {
        features: [5.0, 0.5, 0.94, 0.8, 4, 4, 1, 0.4],
        scenario: "Well-balanced business with multiple strengths",
        expectedOutcome: "good"
    },
    {
        features: [25.0, 2.2, 0.76, 0.2, 5, 8, 7, 0.9],
        scenario: "High revenue but poor risk management",
        expectedOutcome: "bad"
    },
    {
        features: [1.8, 0.4, 0.96, 0.4, 6, 3, 0, 0.3],
        scenario: "Small but excellently managed business",
        expectedOutcome: "good"
    },
    {
        features: [12.0, 1.1, 0.91, 0.15, 3, 7, 5, 0.8],
        scenario: "Growing business with cash flow issues",
        expectedOutcome: "bad"
    },
    {
        features: [3.0, 0.7, 0.93, 1.5, 5, 5, 2, 0.5],
        scenario: "Conservative mid-sized business",
        expectedOutcome: "good"
    },
    {
        features: [7.0, 1.8, 0.79, 0.1, 8, 6, 6, 0.85],
        scenario: "Established business showing decline",
        expectedOutcome: "bad"
    },
    {
        features: [9.0, 1.0, 0.97, 0.7, 8, 6, 4, 0.65],
        scenario: "Strong payment history",
        expectedOutcome: "good"
    },
    {
        features: [15.0, 2.5, 0.88, 0.3, 7, 8, 5, 0.7],
        scenario: "Fast-growing business with weak risk management",
        expectedOutcome: "bad"
    },
    {
        features: [4.0, 0.6, 0.94, 0.5, 6, 5, 3, 0.6],
        scenario: "Small business with recent growth",
        expectedOutcome: "good"
    },
    {
        features: [18.0, 1.9, 0.85, 0.1, 8, 7, 6, 0.7],
        scenario: "Established business with cash flow issues",
        expectedOutcome: "bad"
    },
    {
        features: [10.0, 1.4, 0.92, 0.75, 5, 4, 3, 0.55],
        scenario: "Random test 1",
        expectedOutcome: "good"
    },
    {
        features: [6.0, 0.8, 0.95, 0.8, 7, 5, 4, 0.7],
        scenario: "Random test 2",
        expectedOutcome: "bad"
    },
    {
        features: [8.0, 1.2, 0.87, 0.6, 6, 7, 5, 0.75],
        scenario: "Random test 3",
        expectedOutcome: "good"
    },
    {
        features: [2.0, 0.9, 0.89, 0.5, 5, 6, 3, 0.8],
        scenario: "Random test 4",
        expectedOutcome: "bad"
    },
    {
        features: [11.0, 1.3, 0.92, 0.6, 7, 6, 5, 0.75],
        scenario: "Random test 5",
        expectedOutcome: "good"
    },
    {
        features: [3.0, 0.8, 0.95, 0.9, 6, 5, 4, 0.7],
        scenario: "Random test 6",
        expectedOutcome: "bad"
    }
];

// Predict probabilities for the test data using predictBatch
const predictionsBatch = model.predictBatch(X_test.map(test => test.features));

// Convert probability scores to credit scores using a scaling function
function probabilityToCreditScore(probability) {
    // Ensure probability is in [0,1]
    const p = Math.max(0, Math.min(1, probability));
    
    // Use sigmoid scaling for more distinction in middle range
    const sigmoid = 1 / (1 + Math.exp(-12 * (p - 0.5)));
    
    // Scale to credit score range
    const minScore = 300;
    const maxScore = 850;
    const score = minScore + (maxScore - minScore) * sigmoid;
    
    return Math.round(score);
}

// Modified output format
console.log('\nCredit Score Predictions:');
console.log('=========================\n');


// Threshold Analysis
const checkThresholds = (features) => {
    const checks = [
        features[0] >= THRESHOLDS.MIN_ANNUAL_REVENUE,
        features[1] <= THRESHOLDS.MAX_DEBT_TO_EQUITY,
        features[2] >= THRESHOLDS.MIN_PAYMENT_HISTORY,
        features[3] >= THRESHOLDS.MIN_CASH_RESERVES,
        features[4] >= THRESHOLDS.MIN_YEARS_IN_BUSINESS,
        features[5] <= THRESHOLDS.MAX_INDUSTRY_RISK,
        features[6] <= THRESHOLDS.MAX_LATE_PAYMENTS,
        features[7] <= THRESHOLDS.MAX_CREDIT_UTILIZATION
    ];
    return checks.filter(Boolean).length;
};

// Add threshold analysis to output
console.log('\nThreshold Analysis:');
console.log('==================');
X_test.forEach((test, index) => {
    const thresholdsMet = checkThresholds(test.features);
    console.log(`Scenario ${index + 1}: ${test.scenario}`);
    console.log(`Thresholds Met: ${thresholdsMet}/8`);
    console.log(`Required for Good Credit: ≥7\n`);
});

// Retrieve feature importance scores
const importance = model.getFeatureImportance();
console.log('Feature Importance:', importance);

// Update feature names for display
const featureNames = [
    'Annual Revenue',
    'Debt to Equity',
    'Payment History',
    'Cash Reserves',
    'Years in Business',
    'Industry Risk',
    'Late Payments',
    'Credit Utilization'
];

// Combine feature names with their importance scores
const featureImportance = featureNames.map((name, index) => ({
    feature: name,
    importance: importance[index]
}));

// Sort features by importance in descending order
featureImportance.sort((a, b) => b.importance - a.importance);

// Display the feature importances
console.log('Feature Importances:');
featureImportance.forEach(({ feature, importance }) => {
    console.log(`${feature}: ${importance}`);
});

// Create training data for neural network with actual credit scores
function calculateBaseScore(features, thresholds, raw = false) {
    const meetsThresholds = [
        features[0] >= thresholds.MIN_ANNUAL_REVENUE,
        features[1] <= thresholds.MAX_DEBT_TO_EQUITY,
        features[2] >= thresholds.MIN_PAYMENT_HISTORY,
        features[3] >= thresholds.MIN_CASH_RESERVES,
        features[4] >= thresholds.MIN_YEARS_IN_BUSINESS,
        features[5] <= thresholds.MAX_INDUSTRY_RISK,
        features[6] <= thresholds.MAX_LATE_PAYMENTS,
        features[7] <= thresholds.MAX_CREDIT_UTILIZATION
    ];
    
    const thresholdsMet = meetsThresholds.filter(Boolean).length;
    // Base score calculation
    const baseScore = 0 + (thresholdsMet / 8) * (800 - 0);
    
    // Adjustment factors
    const revenueBonus = features[0] > thresholds.MIN_ANNUAL_REVENUE * 2 ? 25 : 0;
    const paymentHistoryBonus = features[2] > 0.95 ? 25 : features[2] > 0.9 ? 15 : 0;
    const riskPenalty = features[5] > thresholds.MAX_INDUSTRY_RISK ? -30 : 0;
    const utilizationPenalty = features[7] > thresholds.MAX_CREDIT_UTILIZATION ? -20 : 0;
    
    return Math.min(850, Math.max(raw === true? 90 : 300, 
        baseScore + revenueBonus + paymentHistoryBonus + riskPenalty + utilizationPenalty
    ));
}

export function calculateStandardizedScore(features) {
    // Base score starts at 300
    let baseScore = 300;
    const maxPoints = 550; // Points available above base score
    
    // Payment Behavior (35% of additional points = 192.5 points)
    const paymentPoints = MODIFIED_SBSS_WEIGHTS.PAYMENT_BEHAVIOR.weight * maxPoints;
    const paymentScore = (
        // Payment history
        (features[2] >= 0.95 ? 1.0 : 
         features[2] >= 0.90 ? 0.8 :
         features[2] >= 0.85 ? 0.6 :
         features[2] >= 0.80 ? 0.4 : 0.2) * 
         (paymentPoints * MODIFIED_SBSS_WEIGHTS.PAYMENT_BEHAVIOR.metrics.paymentHistory) +
        
        // Late payments 
        ((THRESHOLDS.MAX_LATE_PAYMENTS - features[6]) / THRESHOLDS.MAX_LATE_PAYMENTS) * 
         (paymentPoints * MODIFIED_SBSS_WEIGHTS.PAYMENT_BEHAVIOR.metrics.latePayments) +
        
        // Credit utilization
        (features[7] <= 0.3 ? 1.0 :
         features[7] <= 0.5 ? 0.8 :
         features[7] <= 0.7 ? 0.6 :
         features[7] <= 0.8 ? 0.4 : 0.2) * 
         (paymentPoints * MODIFIED_SBSS_WEIGHTS.PAYMENT_BEHAVIOR.metrics.creditUtilization) 
    );

    // Business Fundamentals (35% of additional points = 192.5 points)
    const businessPoints = MODIFIED_SBSS_WEIGHTS.BUSINESS_FUNDAMENTALS.weight * maxPoints;
    const businessScore = (
        // Annual revenue 
        (features[0] >= THRESHOLDS.MIN_ANNUAL_REVENUE * 5 ? 1.0 :
         features[0] >= THRESHOLDS.MIN_ANNUAL_REVENUE * 3 ? 0.8 :
         features[0] >= THRESHOLDS.MIN_ANNUAL_REVENUE * 2 ? 0.6 :
         features[0] >= THRESHOLDS.MIN_ANNUAL_REVENUE ? 0.4 : 0.2) * 
         (businessPoints * MODIFIED_SBSS_WEIGHTS.BUSINESS_FUNDAMENTALS.metrics.annualRevenue) +
        
        // Cash reserves 
        (features[3] >= THRESHOLDS.MIN_CASH_RESERVES * 3 ? 1.0 :
         features[3] >= THRESHOLDS.MIN_CASH_RESERVES * 2 ? 0.8 :
         features[3] >= THRESHOLDS.MIN_CASH_RESERVES * 1.5 ? 0.6 :
         features[3] >= THRESHOLDS.MIN_CASH_RESERVES ? 0.4 : 0.2) * 
         (businessPoints * MODIFIED_SBSS_WEIGHTS.BUSINESS_FUNDAMENTALS.metrics.cashReserves) +
        
        // Debt to equity 
        (features[1] <= THRESHOLDS.MAX_DEBT_TO_EQUITY * 0.5 ? 1.0 :
         features[1] <= THRESHOLDS.MAX_DEBT_TO_EQUITY * 0.75 ? 0.8 :
         features[1] <= THRESHOLDS.MAX_DEBT_TO_EQUITY ? 0.6 :
         features[1] <= THRESHOLDS.MAX_DEBT_TO_EQUITY * 1.5 ? 0.4 : 0.2) * 
         (businessPoints * MODIFIED_SBSS_WEIGHTS.BUSINESS_FUNDAMENTALS.metrics.debtToEquity)
    );

    // Risk Factors (30% of additional points = 165 points)
    const riskPoints = MODIFIED_SBSS_WEIGHTS.RISK_FACTORS.weight * maxPoints;
    const riskScore = (
        // Industry risk 
        (features[5] <= THRESHOLDS.MAX_INDUSTRY_RISK * 0.5 ? 1.0 :
         features[5] <= THRESHOLDS.MAX_INDUSTRY_RISK * 0.75 ? 0.8 :
         features[5] <= THRESHOLDS.MAX_INDUSTRY_RISK ? 0.6 :
         features[5] <= THRESHOLDS.MAX_INDUSTRY_RISK * 1.25 ? 0.4 : 0.2) * 
         (riskPoints * MODIFIED_SBSS_WEIGHTS.RISK_FACTORS.metrics.industryRisk) +
        
        // Years in business 
        (features[4] >= THRESHOLDS.MIN_YEARS_IN_BUSINESS * 3 ? 1.0 :
         features[4] >= THRESHOLDS.MIN_YEARS_IN_BUSINESS * 2 ? 0.8 :
         features[4] >= THRESHOLDS.MIN_YEARS_IN_BUSINESS * 1.5 ? 0.6 :
         features[4] >= THRESHOLDS.MIN_YEARS_IN_BUSINESS ? 0.4 : 0.2) * 
         (riskPoints * MODIFIED_SBSS_WEIGHTS.RISK_FACTORS.metrics.yearsInBusiness) +
        
        // Credit history using payment history
        (features[2] >= 0.95 ? 1.0 :
         features[2] >= 0.90 ? 0.8 :
         features[2] >= 0.85 ? 0.6 :
         features[2] >= 0.80 ? 0.4 : 0.2) * 
         (riskPoints * MODIFIED_SBSS_WEIGHTS.RISK_FACTORS.metrics.creditHistory)
    );

    // Debug logging
    // console.log(`Score Components:
    //     Payment Score: ${paymentScore.toFixed(2)} / 192.50
    //     Business Score: ${businessScore.toFixed(2)} / 192.50
    //     Risk Score: ${riskScore.toFixed(2)} / 165.00
    //     Total Additional Points: ${(paymentScore + businessScore + riskScore).toFixed(2)} / 550
    // `);

    const finalScore = Math.round(baseScore + paymentScore + businessScore + riskScore);
    return Math.min(850, Math.max(300, finalScore));
}


// Function to normalize features
export function normalizeFeatures(features) {
    return features.map((value, index) =>
        Math.min(1, Math.max(0, value / FEATURE_RANGES[index][1]))
    );
}


// Function to denormalize credit score
function denormalizeCreditScore(normalizedScore) {
    return Math.round(0 + normalizedScore * (800 - 0));
}

// Normalize training data before creating neural network training set
const normalizedX_train = processedX_train.map(features => 
    normalizeFeatures(features)
);

// Prepare neural network training data
const nnTrainingData = normalizedX_train.map((normalizedFeatures, index) => {
    const baseScore = calculateBaseScore(processedX_train[index], THRESHOLDS, true);
    // Normalize the credit score to [0,1] range for training
    const normalizedScore = (baseScore - 0) / (800 - 0);
    return {
        input: normalizedFeatures,
        output: [normalizedScore]
    };
});

// Convert binary scores to credit scores using base score calculation
const creditScores = processedX_train.map(features => 
    calculateBaseScore(features, THRESHOLDS, true)
);
const creditScoresFICO = processedX_train.map(features => 
    calculateStandardizedScore(features)//, THRESHOLDS, false)
);

//NEURAL NETWORK MODELS
// Initialize models
const nn = new NeuralNetwork(
    8,              // input size (number of features)
    [16, 10],        // hidden layers
    1,              // output size (credit score)
    'swish'         // activation function
);
const regressor = new CreditRegressor();
const ordinalClassifier = new OrdinalCreditClassifier();



// Train all models
console.log('\nTraining Generic Neural Network...');
nn.train(nnTrainingData, 0.001, 3000, true);
console.log('\nTraining Regressor Neural Network...');
regressor.train(processedX_train, creditScoresFICO);
console.log('\nTraining Ordinal Classifier Neural Network...');
ordinalClassifier.train(processedX_train, creditScoresFICO);

// Initialize interest rate model
console.log('\nSetting Up Interest Rate Model...');
const interestModel = new InterestRateModel();

//MODELS PREDICTIONS
console.log('\nComparing Models Predictions:');
console.log('============================\n');
// Test predictions
X_test.forEach((test, index) => {
    
    const xgbProbability = predictionsBatch[index];
    const creditScore = probabilityToCreditScore(xgbProbability);
    const features = test.features;
    const xgbScore = probabilityToCreditScore(xgbProbability);
    
    // Get neural network prediction
    const normalizedFeatures = normalizeFeatures(test.features);
    const normalizedPrediction = nn.forward(normalizedFeatures)[0];
    const nnScore = denormalizeCreditScore(normalizedPrediction);

    //NN Models Predictions
    const regressionResult = regressor.predict(test.features);
    const ordinalResult = ordinalClassifier.predict(test.features);

    //Ensemble Reults
    const ficoScore = calculateStandardizedScore(features);


        // Prepare features for interest rate model
        const interestRateFeatures = {
            xgboostScore: xgbProbability,
            nnScore: nnScore,
            regressionScore: regressionResult.score,
            ordinalScore: ordinalResult.score,
            ficoScore: ficoScore,
            paymentHistory: features[2],
            creditUtilization: features[7],
            cashReserves: features[3],
            debtToEquity: features[1],
            industryRisk: features[5] 
        };


    // Get interest rate prediction
    const rateResult = interestModel.predict(interestRateFeatures);

    // Display results
    console.log(`\nTest Scenario ${index + 1}: ${test.scenario}`);

    console.log('Business Metrics:');
    console.log(`- Annual Revenue: KES ${features[0]}M`);
    console.log(`- Debt to Equity: ${features[1]}`);
    console.log(`- Payment History: ${(features[2] * 100).toFixed(1)}%`);
    console.log(`- Cash Reserves: KES ${features[3]}M`);
    console.log(`- Years in Business: ${features[4]}`);
    console.log(`- Industry Risk Score: ${features[5]}/10`);
    console.log(`- Late Payments: ${features[6]}`);
    console.log(`- Credit Utilization: ${(features[7] * 100).toFixed(1)}%`);

    console.log(`\nQuick Classification: ${test.expectedOutcome.toUpperCase()}`);
    console.log(`\nExpected Outcome (FICO SBSS Standardized Classification): ${calculateStandardizedScore(features)}`);
    console.log(`- Expected Category: ${calculateStandardizedScore(features) >= 680 ? 'GOOD' : 'BAD'}`);
    console.log('\nModel Predictions: (Threshold: 680)');
    console.log('=====================');
    console.log(`XGBoost: `);
    console.log(`- Category: ${xgbProbability >= 0.5 ? 'GOOD' : 'BAD'}`);
    console.log(`- Probability: ${(xgbProbability * 100).toFixed(1)}%`);

    console.log(`\nNeural Network: ${nnScore > 850 ? 850 : nnScore}` );
    console.log(`- Score: ${nnScore > 850 ? 850 : nnScore}`);
    console.log(`- Category: ${nnScore >= 680 ? 'GOOD' : 'BAD'} `);
    console.log(`Raw NN Output: ${(normalizedPrediction).toFixed(4)}`);

    console.log('\nNeural Network Regression with Binning:');
    console.log(`- Raw Score: ${regressionResult.score}`);
    console.log(`- Category: ${regressionResult.category}`);
    console.log(`- Range: ${regressionResult.range}`);
    
    console.log('\nNeural Network Ordinal Classification:');
    console.log(`- Score: ${ordinalResult.score}`);
    console.log(`- Category: ${ordinalResult.category}`);
    console.log(`- Range: ${ordinalResult.range}`);
    console.log(`- Raw Predictions: ${ordinalResult.predictions}`);
    console.log(`- Confidence: ${(ordinalResult.confidence * 100).toFixed(1)}%`);
    console.log(`\nFINAL CREDIT SCORE: ${calculateFinalCreditScore(nnScore, regressionResult.score, ordinalResult.score, ficoScore, xgbProbability)}`); //(0.995 * (nnScore + regressionResult.score + ordinalResult.score) / 3).toFixed(0)}`);

    // Add interest rate to output
    console.log('\nInterest Rate Analysis:');
    console.log(`- Base Rate: ${rateResult.baseRate}`);
    console.log(`- Final Rate: ${rateResult.adjustedRate}`);
    console.log(`- Risk Multiplier: ${rateResult.riskMultiplier}`);
    if (rateResult.riskFactors.length > 0) {
        console.log('- Risk Factors:');
        rateResult.riskFactors.forEach(factor => 
            console.log(`  * ${factor}`)
        );
    }
    console.log('\n' + '='.repeat(50));
});

// Weghted Ensemble
export function calculateFinalCreditScore(nnScore, regressionNNScore, ordinalNNScore, ficoScore, xgbScore) {
    const threshold = 680;
    // Calculate absolute differences from ficoScore
    const diffNN = Math.abs(nnScore - ficoScore);
    const diffReg = Math.abs(regressionNNScore - ficoScore);
    const diffOrd = Math.abs(ordinalNNScore - ficoScore);
    
    // Calculate weights (inverse of differences, with small epsilon to avoid division by zero)
    const epsilon = 1e-6;
    const weightNN = 1 / (diffNN + epsilon);
    const weightReg = 1 / (diffReg + epsilon);
    const weightOrd = 1 / (diffOrd + epsilon);
    
    // Normalize weights to sum to 1
    const totalWeight = weightNN + weightReg + weightOrd;
    const normalizedWeightNN = weightNN / totalWeight;
    const normalizedWeightReg = weightReg / totalWeight;
    const normalizedWeightOrd = weightOrd / totalWeight;
    
    // Calculate weighted average
    const weightedAverage = (nnScore * normalizedWeightNN +
                            regressionNNScore * normalizedWeightReg +
                            ordinalNNScore * normalizedWeightOrd);

    // Return ficoScore if conditions are met
    if (xgbScore > 0.5 && weightedAverage < threshold && ficoScore > threshold) {
        return ficoScore.toFixed(0) > threshold + 10 ? threshold + 10 : ficoScore.toFixed(0);
    }

    // Return double weighted score if conditions are met
    if (xgbScore < 0.5 && weightedAverage < threshold && ficoScore > threshold) {
        return ( 0.975 * ((weightedAverage + ficoScore)/2)).toFixed(0);
    }
    
    // Return weighted average
    return weightedAverage.toFixed(0);
}

function XcalculateFinalCreditScore(nnScore, regressionNNScore, ordinalNNScore, ficoScore, xgbScore) {
    return ((nnScore + regressionNNScore + ordinalNNScore)/3).toFixed(0);
}

/**
 * Makes predictions using all available models and returns results in API-friendly JSON format
 * @param {Array} features - Array of 8 feature values in order:
 *    [annualRevenue, debtToEquity, paymentHistory, cashReserves,
 *     yearsInBusiness, industryRisk, latePayments, creditUtilization]
 * @returns {Object} Prediction results from all models in consolidated format
 */
export function predictAll(features) {
    // Get XGBoost prediction
    const xgbProbability = model.predictSingle(features);
    const xgbScore = probabilityToCreditScore(xgbProbability);

    // Get neural network prediction
    const normalizedFeatures = normalizeFeatures(features);
    const normalizedPrediction = nn.forward(normalizedFeatures)[0];
    const nnScore = denormalizeCreditScore(normalizedPrediction);

    // Get other model predictions
    const regressionResult = regressor.predict(features);
    const ordinalResult = ordinalClassifier.predict(features);

    //OG prediction
    const ficoScore = calculateStandardizedScore(features);

    // Prepare features for interest rate model
    const interestRateFeatures = {
        xgboostScore: xgbProbability,
        nnScore: nnScore,
        regressionScore: regressionResult.score,
        ordinalScore: ordinalResult.score,
        ficoScore: ficoScore,
        paymentHistory: features[2],
        creditUtilization: features[7],
        cashReserves: features[3],
        debtToEquity: features[1],
        industryRisk: features[5]
    };

    // Get interest rate prediction
    const rateResult = interestModel.predict(interestRateFeatures);

    // Return consolidated results
    return {
        features: {
            annualRevenue: features[0],
            debtToEquity: features[1],
            paymentHistory: features[2],
            cashReserves: features[3],
            yearsInBusiness: features[4],
            industryRisk: features[5],
            latePayments: features[6],
            creditUtilization: features[7]
        },
        predictions: {
            xgboost: {
                probability: xgbProbability,
                //score: xgbScore,
                category: xgbProbability >= 0.5 ? 'GOOD' : 'BAD'
            },
            neuralNetwork: {
                score: nnScore > 850 ? 850 : nnScore,
                category: nnScore >= 680 ? 'GOOD' : 'BAD',
                rawOutput: normalizedPrediction
            },
            regression: {
                score: regressionResult.score,
                category: regressionResult.category,
                range: regressionResult.range
            },
            ordinalClassification: {
                score: ordinalResult.score,
                category: ordinalResult.category,
                range: ordinalResult.range,
                //confidence: ordinalResult.confidence,
                //rawPredictions: ordinalResult.predictions
            },
            interestRate: {
                baseRate: rateResult.baseRate,
                finalRate: rateResult.adjustedRate,
                riskMultiplier: rateResult.riskMultiplier,
                riskFactors: rateResult.riskFactors
            },
            finalScore: calculateFinalCreditScore(nnScore, regressionResult.score, ordinalResult.score, ficoScore, xgbProbability)
        },
        thresholdsMet: checkThresholds(features),
        standardizedFICOScore: calculateStandardizedScore(features)
    };
}

// API Server Configuration
Bun.serve({
  port: 2226,
  routes: {
    // Individual model prediction endpoints
    "/predict/xgboost": {
      POST: async (req) => {
        try {
          const body = await req.json();
          if (!body.features || !Array.isArray(body.features)) {
            return Response.json(
              { error: "Features array is required" },
              { status: 400 }
            );
          }
          
          if (body.features.length !== 8) {
            return Response.json(
              { error: "Exactly 8 features are required" },
              { status: 400 }
            );
          }
          
          const probability = model.predictSingle(body.features);
          return Response.json({
            success: true,
            prediction: {
              probability,
              category: probability >= 0.5 ? 'GOOD' : 'BAD'
            }
          });
          
        } catch (error) {
          return Response.json(
            { error: error.message },
            { status: 500 }
          );
        }
      }
    },
    
    "/predict/neuralNetwork": {
      POST: async (req) => {
        try {
          const body = await req.json();
          if (!body.features || !Array.isArray(body.features)) {
            return Response.json(
              { error: "Features array is required" },
              { status: 400 }
            );
          }
          
          if (body.features.length !== 8) {
            return Response.json(
              { error: "Exactly 8 features are required" },
              { status: 400 }
            );
          }
          
          const normalizedFeatures = normalizeFeatures(body.features);
          const normalizedPrediction = nn.forward(normalizedFeatures)[0];
          const score = denormalizeCreditScore(normalizedPrediction);
          
          return Response.json({
            success: true,
            prediction: {
              score: score > 850 ? 850 : score,
              category: score >= 680 ? 'GOOD' : 'BAD',
              rawOutput: normalizedPrediction
            }
          });
          
        } catch (error) {
          return Response.json(
            { error: error.message },
            { status: 500 }
          );
        }
      }
    },
    
    "/predict/regression": {
      POST: async (req) => {
        try {
          const body = await req.json();
          if (!body.features || !Array.isArray(body.features)) {
            return Response.json(
              { error: "Features array is required" },
              { status: 400 }
            );
          }
          
          const result = regressor.predict(body.features);
          return Response.json({
            success: true,
            prediction: {
              score: result.score,
              category: result.category,
              range: result.range
            }
          });
          
        } catch (error) {
          return Response.json(
            { error: error.message },
            { status: 500 }
          );
        }
      }
    },
    
    "/predict/ordinalClassification": {
      POST: async (req) => {
        try {
          const body = await req.json();
          if (!body.features || !Array.isArray(body.features)) {
            return Response.json(
              { error: "Features array is required" },
              { status: 400 }
            );
          }
          
          const result = ordinalClassifier.predict(body.features);
          return Response.json({
            success: true,
            prediction: {
              score: result.score,
              category: result.category,
              range: result.range
            }
          });
          
        } catch (error) {
          return Response.json(
            { error: error.message },
            { status: 500 }
          );
        }
      }
    },
    
    // Original consolidated prediction endpoint
    "/predict": {
      POST: async (req) => {
        try {
          const body = await req.json();
          
          // Validate input
          if (!body.features || !Array.isArray(body.features)) {
            return Response.json(
              { error: "Features array is required" },
              { status: 400 }
            );
          }
          
          if (body.features.length !== 8) {
            return Response.json(
              { error: "Exactly 8 features are required" },
              { status: 400 }
            );
          }
          
          // Make prediction
          const prediction = predictAll(body.features);
          
          return Response.json({
            success: true,
            prediction
          });
          
        } catch (error) {
          return Response.json(
            { error: error.message },
            { status: 500 }
          );
        }
      }
    },

    "/add-training-data": {
      POST: async (req) => {
        try {
          const body = await req.json();
          
          // Validate input
          if (!body.features || !Array.isArray(body.features)) {
            return Response.json(
              { error: "Features array is required" },
              { status: 400 }
            );
          }
          
          if (body.features.length !== 8) {
            return Response.json(
              { error: "Exactly 8 features are required" },
              { status: 400 }
            );
          }

          // Read existing training data
          const trainingData = JSON.parse(
            await fs.readFile(path.join(process.cwd(), 'data', 'trainingset.json'), 'utf8')
          );

          // Append new features
          trainingData.push(body.features);

          // Write back to file
          await fs.writeFile(
            path.join(process.cwd(), 'data', 'trainingset.json'),
            JSON.stringify(trainingData, null, 2)
          );

          return Response.json({
            success: true,
            message: "Training data added successfully"
          });
          
        } catch (error) {
          return Response.json(
            { error: error.message },
            { status: 500 }
          );
        }
      }
    },

    "/update": {
      POST: async () => {
        try {
          const result = await updateNNModels();
          return Response.json({
            success: true,
            message: "Neural network models updated successfully",
            result: {
              trainingDataSize: result.trainingData.length
            }
          });
        } catch (error) {
          return Response.json(
            { error: error.message },
            { status: 500 }
          );
        }
      }
    },

    // Serve panel.html
    "/panel": {
        GET: async () => {
        try {
            const html = await fs.readFile(path.join(process.cwd(), 'panel.html'), 'utf8');
            return new Response(html, {
            headers: { 'Content-Type': 'text/html' }
            });
        } catch (error) {
            return new Response("Error loading panel", { status: 500 });
        }
        }
    },
    
    // Health check endpoint
    "/health": new Response("OK"),
    
    // 404 handler for unmatched API routes
    "/*": Response.json({ message: "Not found" }, { status: 404 }),
  },
  

  
  // Fallback for non-API routes
  fetch(req) {
    return new Response("Not Found", { status: 404 });
  }
});

console.log(`\n\nCredit scoring API running on port 2226`);
console.log(`Send request with features array to get predictions:
POST /predict
{
  "features": [5.0, 0.5, 0.94, 0.8, 4, 4, 1, 0.4]
}`);

/**
 * Updates all neural network models with latest training data
 * Loads training data, prepares features, and trains:
 * - Generic neural network (using base score)
 * - Regressor model (using standardized score)
 * - Ordinal classifier (using standardized score)
 */
export async function updateNNModels() {
    try {
        // Load training data
        const data = await fs.readFile(path.join(process.cwd(), 'data', 'trainingset.json'), 'utf8');
        const processedX_train = JSON.parse(data);
        

        // Prepare data for generic neural network (base score)
        const normalizedX_train = processedX_train.map(features =>
            normalizeFeatures(features)
        );
        const nnTrainingData = normalizedX_train.map((normalizedFeatures, index) => {
            const baseScore = calculateBaseScore(processedX_train[index], THRESHOLDS, true);
            const normalizedScore = (baseScore - 0) / (800 - 0);
            return {
                input: normalizedFeatures,
                output: [normalizedScore]
            };
        });

        // Prepare data for regressor/ordinal models (standardized score)
        const creditScoresFICO = processedX_train.map(features =>
            calculateStandardizedScore(features)
        );

        // Initialize models
        //const nn = new NeuralNetwork(8, [16, 10], 1, 'swish');
        //const regressor = new CreditRegressor();
        //const ordinalClassifier = new OrdinalCreditClassifier();

        // Train models
        console.log('\nTraining Generic Neural Network...');
        nn.train(nnTrainingData, 0.001, 3000, true);
        
        console.log('\nTraining Regressor Neural Network...');
        regressor.train(processedX_train, creditScoresFICO);
        
        console.log('\nTraining Ordinal Classifier Neural Network...');
        ordinalClassifier.train(processedX_train, creditScoresFICO);

        // Run tests on X_test
        console.log('\nRunning Tests on Updated Models:');
        
        X_test.forEach((test, index) => {
            const features = test.features;
            const xgbProbability = model.predictSingle(features);
            const normalizedFeatures = normalizeFeatures(test.features);
            const normalizedPrediction = nn.forward(normalizedFeatures)[0];
            const nnScore = denormalizeCreditScore(normalizedPrediction);
            
            const regressionResult = regressor.predict(test.features);
            const ordinalResult = ordinalClassifier.predict(test.features);
            const ficoScore = calculateStandardizedScore(test.features);

            
            console.log('============================\n');
            console.log('============================');
            console.log(`Test ${index + 1}: ${test.scenario}`);
            console.log('Business Metrics:');
            console.log(`- Annual Revenue: KES ${features[0]}M`);
            console.log(`- Debt to Equity: ${features[1]}`);
            console.log(`- Payment History: ${(features[2] * 100).toFixed(1)}%`);
            console.log(`- Cash Reserves: KES ${features[3]}M`);
            console.log(`- Years in Business: ${features[4]}`);
            console.log(`- Industry Risk Score: ${features[5]}/10`);
            console.log(`- Late Payments: ${features[6]}`);
            console.log(`- Credit Utilization: ${(features[7] * 100).toFixed(1)}%`);
            console.log(`\nResults: `);
            console.log(`- XGBoost Category: ${xgbProbability >= 0.5 ? 'GOOD' : 'BAD'}`);
            console.log(`- NN Score: ${nnScore}`);
            console.log(`- Regressor Score: ${regressionResult.score}`);
            console.log(`- Ordinal Category: ${ordinalResult.category}`);
            console.log(`- FICO Score: ${ficoScore}`);
            console.log(`\nFINAL CREDIT SCORE: ${calculateFinalCreditScore(nnScore, regressionResult.score, ordinalResult.score, ficoScore, xgbProbability)}`); //(0.995 * (nnScore + regressionResult.score + ordinalResult.score) / 3).toFixed(0)}`);

        });

        return {
            nn,
            regressor,
            ordinalClassifier,
            trainingData: processedX_train
        };
    } catch (error) {
        console.error('Error updating models:', error);
        throw error;
    }
}