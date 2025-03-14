// Move Node class outside
class Node {
  constructor() {
    this.featureIndex = null;
    this.threshold = null;
    this.left = null;
    this.right = null;
    this.value = null;
    this.isLeaf = false;
  }
}

class XGBoost {
  constructor(params = {}) {
    this.learningRate = params.learningRate || 0.3;
    this.maxDepth = params.maxDepth || 4;
    this.minChildWeight = params.minChildWeight || 1;
    this.numRounds = params.numRounds || 100;
    this.objective = params.objective || 'binary';
    this.trees = [];
    this.y = [];
    this.numFeatures = 0;
  }

  fit(X, y) {
    this.y = y;
    this.numFeatures = X[0].length;

    // Add input validation
    if (!Array.isArray(X) || !Array.isArray(y) || X.length === 0 || y.length === 0) {
      throw new Error('Invalid input data');
    }

    // Initialize predictions based on objective
    let predictions = new Array(y.length).fill(
      this.objective === 'regression' 
        ? y.reduce((a, b) => a + b, 0) / y.length 
        : 0.5
    );
    
    // Add try-catch for error handling
    try {
      for (let i = 0; i < this.numRounds; i++) {
        // Calculate gradients based on objective
        const gradients = y.map((actual, idx) => {
          const pred = predictions[idx];
          if (this.objective === 'regression') {
            // Modified gradient calculation for regression
            return (actual - Math.max(0, Math.min(1, pred))) * 0.1;
          } else {
            const prob = 1 / (1 + Math.exp(-pred));
            return actual - prob;  // Log loss gradient for binary
          }
        });

        // Build tree using sorted indices array
        const sortedIndicesArray = this._getSortedIndices(X);
        const tree = this._buildTree(X, gradients, 0, sortedIndicesArray);
        if (!tree) continue; // Skip if tree building fails
        this.trees.push({ root: tree });

        // Update predictions
        predictions = predictions.map((pred, idx) => {
          const update = this._predict(X[idx], tree.root);
          return pred + this.learningRate * update;
        });
      }
    } catch (error) {
      console.error('Error during training:', error);
      throw error;
    }
  }

  _getSortedIndices(X) {
    const numFeatures = X[0].length;
    const sortedIndices = [];
    
    for (let f = 0; f < numFeatures; f++) {
      sortedIndices.push(
        Array.from({length: X.length}, (_, i) => i)
          .sort((a, b) => X[a][f] - X[b][f])
      );
    }
    return sortedIndices;
  }

  _buildTree(X, gradients, depth, sortedIndicesArray) {
    const node = new Node();
    
    // Validate input
    if (!X || !X.length || !X[0] || !gradients || !sortedIndicesArray) {
      node.isLeaf = true;
      node.value = 0;
      return node;
    }
    
    const sumGrad = gradients.reduce((a, b) => a + b, 0);
    const count = gradients.length;
    
    // Early stopping conditions
    if (depth >= this.maxDepth || 
        count < this.minChildWeight || 
        Math.abs(sumGrad) < 1e-10 || 
        X.length < 2) {
      node.isLeaf = true;
      node.value = sumGrad / (count + 1e-10);
      return node;
    }

    let bestGain = 0;
    let bestSplit = null;

    // Try splitting on each feature
    for (let featureIdx = 0; featureIdx < X[0].length; featureIdx++) {
      if (!sortedIndicesArray[featureIdx]) continue;
      
      let leftSum = 0;
      let leftCount = 0;
      let rightSum = sumGrad;
      let rightCount = count;

      // Evaluate split points
      for (let i = 0; i < sortedIndicesArray[featureIdx].length - 1; i++) {
        const currentIdx = sortedIndicesArray[featureIdx][i];
        const nextIdx = sortedIndicesArray[featureIdx][i + 1];
        
        // Skip if indices are invalid
        if (!X[currentIdx] || !X[nextIdx]) continue;
        
        leftSum += gradients[currentIdx];
        rightSum -= gradients[currentIdx];
        leftCount++;
        rightCount--;

        // Skip identical feature values
        if (X[currentIdx][featureIdx] === X[nextIdx][featureIdx]) {
          continue;
        }

        const gain = (leftSum * leftSum) / (leftCount + 1e-10) + 
                    (rightSum * rightSum) / (rightCount + 1e-10) - 
                    (sumGrad * sumGrad) / (count + 1e-10);

        if (gain > bestGain) {
          bestGain = gain;
          bestSplit = {
            featureIndex: featureIdx,
            threshold: (X[currentIdx][featureIdx] + X[nextIdx][featureIdx]) / 2,
            leftIndices: sortedIndicesArray[featureIdx].slice(0, i + 1),
            rightIndices: sortedIndicesArray[featureIdx].slice(i + 1)
          };
        }
      }
    }

    // If no valid split found, make a leaf
    if (!bestSplit || bestGain < 1e-10) {
      node.isLeaf = true;
      node.value = sumGrad / (count + 1e-10);
      return node;
    }

    // Create child nodes
    node.featureIndex = bestSplit.featureIndex;
    node.threshold = bestSplit.threshold;

    const leftX = bestSplit.leftIndices.map(i => X[i]);
    const leftGrad = bestSplit.leftIndices.map(i => gradients[i]);
    node.left = this._buildTree(leftX, leftGrad, depth + 1, sortedIndicesArray);

    const rightX = bestSplit.rightIndices.map(i => X[i]);
    const rightGrad = bestSplit.rightIndices.map(i => gradients[i]);
    node.right = this._buildTree(rightX, rightGrad, depth + 1, sortedIndicesArray);

    return node;
  }

  predictSingle(x) {
    if (!this.trees.length) return 0;

    let pred = this.objective === 'regression'
      ? this.y.reduce((a, b) => a + b, 0) / this.y.length
      : 0.5;
    
    for (const tree of this.trees) {
      pred += this.learningRate * this._predict(x, tree.root);
    }
    
    if (this.objective === 'regression') {
      return Math.max(0, Math.min(1, pred)); // Clamp to valid range
    } else {
      return 1 / (1 + Math.exp(-pred));
    }
  }

  predictBatch(X) {
    return X.map(x => this.predictSingle(x));
  }

  _predict(x, node) {
    // Add null check
    if (!node) {
      return 0;
    }

    if (node.isLeaf) {
      return node.value;
    }
    
    // Add null checks for child nodes
    if (x[node.featureIndex] <= node.threshold) {
      return this._predict(x, node.left) || 0;
    } else {
      return this._predict(x, node.right) || 0;
    }
  }

  getFeatureImportance() {
    const numFeatures = this.numFeatures;
    const importance = new Array(numFeatures).fill(0);
    
    for (const tree of this.trees) {
        this._traverseTreeForImportance(tree.root, importance);
    }
    
    // Normalize importance scores
    const total = importance.reduce((sum, val) => sum + (val || 0), 0);
    return importance.map(val => total > 0 ? (val || 0) / total * 100 : 0);
  }

  _traverseTreeForImportance(node, importance) {
    if (!node || node.isLeaf) return;
    
    // Ensure featureIndex is valid
    if (node.featureIndex >= 0 && node.featureIndex < importance.length) {
        importance[node.featureIndex] = (importance[node.featureIndex] || 0) + 1;
    }
    
    this._traverseTreeForImportance(node.left, importance);
    this._traverseTreeForImportance(node.right, importance);
  }

  toJSON() {
    return {
        trees: this.trees,
        params: {
            learningRate: this.learningRate,
            maxDepth: this.maxDepth,
            minChildWeight: this.minChildWeight,
            numRounds: this.numRounds
        }
    };
  }

  static fromJSON(json) {
    const model = new XGBoost(json.params);
    model.trees = json.trees;
    return model;
  }
}

// Add this export statement at the end
module.exports = { XGBoost };
