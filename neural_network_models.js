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
            [32, 16, 8],      // hidden layers
            7,              // output (one per category)
            'swish'
        );
    }

    train(features, scores) {
        // Convert scores to ordinal labels with boundary handling
        const ordinalLabels = scores.map(score => {
            // Handle scores below minimum range
            if (score < CREDIT_BINS[0].min) {
                return 0; // Very Poor
            }
            // Handle scores above maximum range
            if (score > CREDIT_BINS[CREDIT_BINS.length - 1].max) {
                return CREDIT_BINS.length - 1; // Exceptional
            }
            
            const bin = CREDIT_BINS.find(b => 
                score >= b.min && score <= b.max
            );
            
            // Fallback if no bin found (shouldn't happen with above checks)
            return bin ? bin.ordinal : 0;
        });

        // Create one-hot encoded targets with validation
        const trainingData = features.map((feature, i) => {
            const ordinal = ordinalLabels[i];
            return {
                input: normalizeFeatures(feature),
                output: Array(7).fill(0).map((_, j) => 
                    j <= ordinal ? 1 : 0
                )
            };
        });

        this.nn.train(trainingData, 0.001, 2000, true);
    }

    predict(features) {
        const normalizedFeatures = normalizeFeatures(features);
        const predictions = this.nn.forward(normalizedFeatures);
        
        // Convert ordinal outputs to credit category with validation
        let ordinal = predictions.findIndex((p, i) => 
            p < 0.5 && (i === 0 || predictions[i-1] >= 0.5)
        ) - 1;

        // Ensure ordinal is within valid range
        ordinal = Math.max(0, Math.min(CREDIT_BINS.length - 1, ordinal));
        
        const bin = CREDIT_BINS[ordinal];
        const midScore = Math.round((bin.max + bin.min) / 2);

        return {
            score: midScore,
            category: bin.label,
            range: `${bin.min}-${bin.max}`,
            confidence: predictions[ordinal]
        };
    }
}

export { CreditRegressor, OrdinalCreditClassifier };