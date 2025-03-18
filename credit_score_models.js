import NeuralNetwork from './neural_network.js';
import { normalizeFeatures, FEATURE_RANGES } from './credit-scorer.js';

const CREDIT_BINS = [
    { min: 300, max: 379, label: 'Very Poor', ordinal: 0 },
    { min: 380, max: 499, label: 'Poor', ordinal: 1 },
    { min: 500, max: 599, label: 'Fair', ordinal: 2 },
    { min: 600, max: 679, label: 'Okay :/', ordinal: 3 },
    { min: 680, max: 739, label: 'Very Good', ordinal: 4 },
    { min: 740, max: 799, label: 'Excellent', ordinal: 5 },
    { min: 800, max: 850, label: 'Exceptional', ordinal: 6 }
];

class CreditRegressor {
    constructor() {
        this.nn = new NeuralNetwork(
            8,              // input features
            [32, 16, 8],   // hidden layers
            1,              // output (credit score)
            'swish'
        );
    }

    train(features, scores) {
        // Normalize scores to [0,1]
        const normalizedScores = scores.map(score => 
            (score - 300) / (850 - 300)
        );

        const trainingData = features.map((feature, i) => ({
            input: normalizeFeatures(feature),
            output: [normalizedScores[i]]
        }));

        this.nn.train(trainingData, 0.001, 2000, true);
    }

    predict(features) {
        const normalizedFeatures = normalizeFeatures(features);
        const normalizedPrediction = this.nn.forward(normalizedFeatures)[0];
        const rawScore = Math.round(300 + normalizedPrediction * (850 - 300));
        
        // Find appropriate bin with fallback
        let bin = CREDIT_BINS.find(bin => 
            rawScore >= bin.min && rawScore <= bin.max
        );

        // Fallback for scores outside ranges
        if (!bin) {
            if (rawScore < CREDIT_BINS[0].min) {
                bin = CREDIT_BINS[0];
            } else {
                bin = CREDIT_BINS[CREDIT_BINS.length - 1];
            }
        }

        return {
            score: Math.max(300, Math.min(850, rawScore)),
            category: bin.label,
            range: `${bin.min}-${bin.max}`
        };
    }
}

class OrdinalCreditClassifier {
    constructor() {
        this.nn = new NeuralNetwork(
            8,              // input features
            [64, 32, 16, 16, 8],      // reduced hidden layers
            7,             // output (one per category)
            'swish'      // keep sigmoid for classification
        );
    }

    train(features, scores) {
        // Convert scores to one-hot encoded vectors
        const trainingData = features.map((feature, i) => {
            const score = scores[i];
            // Find the appropriate bin
            const bin = CREDIT_BINS.find(b => 
                score >= b.min && score <= b.max
            ) || (score < CREDIT_BINS[0].min ? CREDIT_BINS[0] : CREDIT_BINS[CREDIT_BINS.length - 1]);
            
            // Create one-hot encoded output
            const output = Array(7).fill(0);
            output[bin.ordinal] = 1; // Set only the correct bin to 1

            return {
                input: normalizeFeatures(feature),
                output: output
            };
        });

        this.nn.train(trainingData, 0.01, 2000, true); // Adjusted learning rate and epochs
    }

    predict(features) {
        const normalizedFeatures = normalizeFeatures(features);
        const predictions = this.nn.forward(normalizedFeatures);
        
        // Find the highest probability category
        const bestOrdinal = predictions.reduce((maxIdx, curr, idx, arr) => 
            curr > arr[maxIdx] ? idx : maxIdx, 0);
        
        const bin = CREDIT_BINS[bestOrdinal];
        const confidence = predictions[bestOrdinal];
        
        // Calculate weighted score within bin range
        const binRange = bin.max - bin.min;
        const baseScore = bin.min + (binRange * confidence);
        
        // Adjust score based on confidence
        const score = Math.round(baseScore);

        return {
            score: Math.min(850, Math.max(300, score)),
            category: bin.label,
            range: `${bin.min}-${bin.max}`,
            confidence: confidence,
            predictions: predictions // Add raw predictions for debugging
        };
    }
}

export { CreditRegressor, OrdinalCreditClassifier };