import NeuralNetwork from './neural_network.js' 
const log = console.log
import asciichart from 'asciichart'

class DQNAgent {
    constructor(env) {
        this.env = env  // Environment instance
        this.stateSize = env.stateSize
        this.actionSize = env.actionSize
        this.gamma = 0.99
        this.epsilon = 1.0
        this.epsilonMin = 0.1
        this.epsilonDecay = 0.99990
        this.learningRate = 0.001
        this.memory = []
        this.scoreCurve = []
        this.temperature = 1.0

        this.model = new NeuralNetwork(this.stateSize, [24,16], this.actionSize)
        this.targetModel = new NeuralNetwork(this.stateSize, [24,16], this.actionSize)
    }

    remember(state, action, reward, nextState, done) {
        const qValues = this.model.forward(state)
        const nextQValues = this.targetModel.forward(nextState)
        const target = reward + (done ? 0 : this.gamma * Math.max(...nextQValues))
        const tdError = Math.abs(target - qValues[action])
    
        // Store with priority (TD error)
        this.memory.push({ state, action, reward, nextState, done, priority: tdError })
    }

    act(state, explore = true) {
        if (explore && Math.random() <= this.epsilon) {
            // Exploration: select a random action
            return Math.floor(Math.random() * this.actionSize);
        }
        // Exploitation: select the action with the highest Q-value (best action)
        const qValues = this.model.forward(state);
        return this.argmax(qValues);
    }
    
    boltAct(state, explore = true) {
        const qValues = this.model.forward(state)
        if (explore && Math.random() <= this.epsilon) {
            // Apply Boltzmann exploration using softmax on Q-values
            const expQValues = qValues.map(q => Math.exp(q / this.temperature));
            const sumExpQ = expQValues.reduce((a, b) => a + b, 0);
            const probabilities = expQValues.map(expQ => expQ / sumExpQ);

            // Select action based on these probabilities
            return this.weightedRandomAction(probabilities);
        }
        // Exploit
        return this.argmax(qValues)
    }

    weightedRandomAction(probabilities) {
        // Generate a random number in [0,1)
        const rand = Math.random();

        // Accumulate probabilities and find the action that matches the random value
        let accumulated = 0;
        for (let i = 0; i < probabilities.length; i++) {
            accumulated += probabilities[i];
            if (rand < accumulated) {
                return i;
            }
        }
        // Fallback, in case of rounding errors
        return probabilities.length - 1;
    }

    argmax(array) {
        return array.indexOf(Math.max(...array))
    }

    replay(batchSize) {
        // Sort memory by priority and sample from the top
        const batch = this.memory.sort((a, b) => b.priority - a.priority).slice(0, batchSize)
        batch.forEach(exp => {
            let target = exp.reward
            if (!exp.done) {
                const nextQ = this.targetModel.forward(exp.nextState)
                target += this.gamma * Math.max(...nextQ)
            }
            const targetQ = this.model.forward(exp.state)
            targetQ[exp.action] = target
            this.model.backward(exp.state, targetQ, this.learningRate)
        });
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay
        }
        if (this.memory.length > 50000) {
            this.memory.shift(); // Remove the oldest experience
        }
    }

    updateTargetModel() {
        //this.targetModel = JSON.parse(JSON.stringify(this.model))
        this.targetModel.weights  = this.model.weights.map(layer => layer.map(node => [...node]))
        this.targetModel.biases = this.model.biases.map(bias => [...bias])
    }

    train(episodes = 100, batchSize = 32, steps = 200) {
        log('Starting training...')
        const startTime = Date.now()
        for (let episode = 0; episode < episodes; episode++) {
            let state = this.env.reset()
            let totalReward = 0

            for (let time = 0; time < steps; time++) {
                const action = this.act(state)
                const { nextState, reward, done } = this.env.step(action)
                this.remember(state, action, reward, nextState, done)

                state = nextState
                totalReward += reward

                if (done) {
                    console.log(`Episode: ${episode + 1}, Score: ${totalReward > 10  ? totalReward.toFixed(1) : totalReward.toFixed(2)} Epsilon: ${this.epsilon.toFixed(3)}`)
                    if(episode % 3 === 2)this.scoreCurve.push(totalReward)
                    break
                } 

                if (this.memory.length >= batchSize) {
                    this.replay(batchSize)
                }

                //console.log(`Episode: ${episode + 1}, Score: ${totalReward} Epsilon: ${this.epsilon}`)
                //if(episode % 10 === 9)this.scoreCurve.push(totalReward)

            }

            this.updateTargetModel()
        }
        const totalTime = (Date.now() - startTime) / 1000
        log('reward:')
        log(asciichart.plot([this.scoreCurve,[0]],{height: 7, colors: [asciichart.blue,asciichart.white]}))
        log(`Training time: ${totalTime} seconds`)
    }

    evaluate(episodes = 10, maxSteps = 200) {
        log('Starting evaluation...')
        let totalRewards = 0
        let rewardHistory = []
        for (let episode = 0; episode < episodes; episode++) {
            let state = this.env.reset()
            let totalReward = 0
            let done = false
            let steps = 0  // Step counter for max step limit
            while (!done && steps < maxSteps) {
                const action = this.act(state, false)  // No exploration
                const { nextState, reward, done: episodeDone } = this.env.step(action)
                state = nextState
                totalReward += reward
                done = episodeDone
                steps++
            }
            // Log if episode terminated early due to max steps
            if (steps >= maxSteps) {
                log(`Episode ${episode + 1} reached maximum step limit.`)
            }
            totalRewards += totalReward
            rewardHistory.push(totalReward)
        }
        log(`Evaluation over ${episodes} episodes: Average score: ${totalRewards / episodes}`)
        log('scores: ',rewardHistory)
    }
    

    
}

export default DQNAgent
