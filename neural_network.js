const log = console.log

class NeuralNetwork {
    constructor(inputSize, hiddenLayers, outputSize, activationType = 'swish') {
        this.inputSize = inputSize;
        this.hiddenLayers = hiddenLayers;
        this.outputSize = outputSize;
        this.activationType = activationType;

        // Initialize weights and biases
        this.weights = [];
        this.biases = [];

        this.weights.push(this.randomMatrix(inputSize, hiddenLayers[0]));
        this.biases.push(Array(hiddenLayers[0]).fill(0));

        for (let i = 1; i < hiddenLayers.length; i++) {
            this.weights.push(this.randomMatrix(hiddenLayers[i - 1], hiddenLayers[i]));
            this.biases.push(Array(hiddenLayers[i]).fill(0));
        }

        this.weights.push(this.randomMatrix(hiddenLayers[hiddenLayers.length - 1], outputSize));
        this.biases.push(Array(outputSize).fill(0));
    }

    randomMatrix(rows, cols) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            const row = [];
            for (let j = 0; j < cols; j++) {
                row.push(Math.random() * Math.sqrt(2 / (rows+cols))); // Small initialization to avoid NaNs
            }
            matrix.push(row);
        }
        return matrix;
    }
    

    relu(x) {
        return x > 0 ? x : 0;
    }

    reluDerivative(x) {
        return x > 0 ? 1 : 0;
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    sigmoidDerivative(x) {
        const sig = this.sigmoid(x);
        return sig * (1 - sig);
    }

    swish(x){
        return x / (1 + Math.exp(-x));
    }

    swishDerivative(x) {
        const sig = this.sigmoid(x);
        return sig + x * sig * (1 - sig);
    }

    activate(x) {
        if (this.activationType === 'relu') {
            return this.relu(x);
        } else if (this.activationType === 'sigmoid') {
            return this.sigmoid(x);
        } else if (this.activationType === 'swish') {
            return this.swish(x);
        }
    }

    activateDerivative(x) {
        if (this.activationType === 'relu') {
            return this.reluDerivative(x);
        } else if (this.activationType === 'sigmoid') {
            return this.sigmoidDerivative(x);
        } else if (this.activationType === 'swish') {
            return this.swishDerivative(x);
        }
    }

    forward(input) {
        this.layerInputs = [];
        this.layerOutputs = [];

        let layerInput = input;
        for (let i = 0; i < this.hiddenLayers.length; i++) {
            const layerOutput = [];
            for (let j = 0; j < this.hiddenLayers[i]; j++) {
                let neuron = this.biases[i][j];
                for (let k = 0; k < layerInput.length; k++) {
                    neuron += layerInput[k] * this.weights[i][k][j];
                }
                layerOutput.push(this.activate(neuron));
            }
            this.layerInputs.push(layerInput);
            this.layerOutputs.push(layerOutput);
            layerInput = layerOutput; // Output of current layer becomes input of next
        }

        // Output layer
        const output = [];
        for (let i = 0; i < this.outputSize; i++) {
            let neuron = this.biases[this.biases.length - 1][i];
            for (let j = 0; j < layerInput.length; j++) {
                neuron += layerInput[j] * this.weights[this.weights.length - 1][j][i];
            }
            output.push(neuron); // No activation if regression
        }
        this.layerInputs.push(layerInput);
        this.layerOutputs.push(output);

        return output
    }

    meanSquaredErrorLoss(target, output) {
        let loss = 0;
        for (let i = 0; i < target.length; i++) {
            loss += (target[i] - output[i]) ** 2;
        }
        console.log("t",target.length)
        return loss / target.length;
    }

    crossEntropyLoss(target, output) {
        let loss = 0;
        for (let i = 0; i < target.length; i++) {
            loss -= target[i] * Math.log(output[i]) + (1 - target[i]) * Math.log(1 - output[i]);
        }
        return loss / target.length;
    }

    huberLoss(target, output) {
        const delta = 1.0;
        let loss = 0;
        for (let i = 0; i < target.length; i++) {
            const error = target[i] - output[i];
            if(Math.abs(error) <= delta){
                loss += 0.5 * error * error; //quadratic loss
            } else {
                loss += delta * Math.abs(error) - 0.5 * delta; //linear loss
            }
        }
        return loss / target.length;

    }

    huberLossDerivative(target, output) {
        const delta = 1.0;
        const error = target - output;
        if(Math.abs(error) <= delta){
            return error;
        } else {
            return (error > 0 ? delta : -delta)
        }
    } //for output layer in backprop calculation only

    backward(input, target, learningRate) {
        const deltas = [];
        const output = this.layerOutputs[this.layerOutputs.length - 1];

        // Calculate delta for output layer
        const outputDelta = [];
        for (let i = 0; i < this.outputSize; i++) {
            outputDelta.push(this.huberLossDerivative(target[i],output[i]));
        }
        deltas.push(outputDelta);

        // Calculate deltas for hidden layers
        for (let i = this.hiddenLayers.length - 1; i >= 0; i--) {
            const layerDelta = [];
            for (let j = 0; j < this.hiddenLayers[i]; j++) {
                let error = 0;
                for (let k = 0; k < (i === this.hiddenLayers.length - 1 ? this.outputSize : this.hiddenLayers[i + 1]); k++) {
                    error += deltas[0][k] * this.weights[i + 1][j][k];
                }
                layerDelta.push(error * this.activateDerivative(this.layerOutputs[i][j]));
            }
            deltas.unshift(layerDelta);
        }

        // Update weights and biases
        for (let i = 0; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    this.weights[i][j][k] += learningRate * deltas[i][k] * this.layerInputs[i][j];
                }
            }
        }

        for (let i = 0; i < this.biases.length; i++) {
            for (let j = 0; j < this.biases[i].length; j++) {
                this.biases[i][j] += learningRate * deltas[i][j];
            }
        }
    }

    train(trainingData, learningRate = 0.01, epochs = 1000, logEpoch = false) {
        const startTime = Date.now();
        const losses = [];
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;
            for (let data of trainingData) {
                const input = data.input;
                const target = data.output;
                const output = this.forward(input);
                
                this.activationType === 'relu' || this.activationType === 'swish' ?
                totalLoss += this.huberLoss(target, output) :
                totalLoss += this.crossEntropyLoss(target, output);
                
                this.backward(input, target, learningRate);
            }

            const averageLoss = totalLoss / trainingData.length;
            losses.push(averageLoss);

            // Log every nth epoch based on total epochs
            if (logEpoch && epoch % Math.max(1, Math.floor(epochs/100)) === 0) {
                log(`Epoch ${epoch + 1}/${epochs}`);
                log(`Loss: ${averageLoss.toFixed(8)}`);
                log(`Progress: ${((epoch + 1)/epochs * 100).toFixed(1)}%`);
                log('-'.repeat(40));
            }
        }

        // Final training summary
        const totalTime = (Date.now() - startTime) / 1000;
        log('\nTraining Summary');
        log('================');
        log(`Initial Loss: ${losses[0].toFixed(8)}`);
        log(`Final Loss: ${losses[losses.length - 1].toFixed(8)}`);
        log(`Improvement: ${((losses[0] - losses[losses.length - 1])/losses[0] * 100).toFixed(2)}%`);
        log(`Training Time: ${totalTime.toFixed(2)} seconds`);
        
        // Log min/max/avg loss
        const minLoss = Math.min(...losses);
        const maxLoss = Math.max(...losses);
        const avgLoss = losses.reduce((a, b) => a + b, 0) / losses.length;
        
        log('\nLoss Statistics');
        log('===============');
        log(`Minimum Loss: ${minLoss.toFixed(8)}`);
        log(`Maximum Loss: ${maxLoss.toFixed(8)}`);
        log(`Average Loss: ${avgLoss.toFixed(8)}`);
    }
}

export default NeuralNetwork;
