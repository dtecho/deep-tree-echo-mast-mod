/**
 * Deep Tree Echo State Network - A hierarchical echo state network implementation
 * This extends the basic ESN to create a tree-like structure with multiple layers
 * and hierarchical processing capabilities.
 */

// Simple matrix implementation to avoid external dependencies
class SimpleMatrix {
  private data: number[][];
  public rows: number;
  public cols: number;

  constructor(data: number[][] | number, cols?: number) {
    if (typeof data === 'number') {
      // Create zero matrix
      this.rows = data;
      this.cols = cols || data;
      this.data = Array(this.rows).fill(null).map(() => Array(this.cols).fill(0));
    } else {
      this.data = data;
      this.rows = data.length;
      this.cols = data[0]?.length || 0;
    }
  }

  static zeros(rows: number, cols: number): SimpleMatrix {
    return new SimpleMatrix(rows, cols);
  }

  static random(rows: number, cols: number): SimpleMatrix {
    const data = Array(rows).fill(null).map(() => 
      Array(cols).fill(null).map(() => Math.random() * 2 - 1)
    );
    return new SimpleMatrix(data);
  }

  static eye(size: number): SimpleMatrix {
    const matrix = SimpleMatrix.zeros(size, size);
    for (let i = 0; i < size; i++) {
      matrix.set(i, i, 1);
    }
    return matrix;
  }

  static columnVector(data: number[]): SimpleMatrix {
    return new SimpleMatrix(data.map(x => [x]));
  }

  static rowVector(data: number[]): SimpleMatrix {
    return new SimpleMatrix([data]);
  }

  get(row: number, col: number): number {
    return this.data[row][col];
  }

  set(row: number, col: number, value: number): void {
    this.data[row][col] = value;
  }

  add(other: SimpleMatrix): SimpleMatrix {
    const result = SimpleMatrix.zeros(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.set(i, j, this.get(i, j) + other.get(i, j));
      }
    }
    return result;
  }

  mul(scalar: number): SimpleMatrix {
    const result = SimpleMatrix.zeros(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.set(i, j, this.get(i, j) * scalar);
      }
    }
    return result;
  }

  mmul(other: SimpleMatrix): SimpleMatrix {
    if (this.cols !== other.rows) {
      throw new Error(`Matrix dimension mismatch: ${this.cols} !== ${other.rows}`);
    }
    
    const result = SimpleMatrix.zeros(this.rows, other.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < other.cols; j++) {
        let sum = 0;
        for (let k = 0; k < this.cols; k++) {
          sum += this.get(i, k) * other.get(k, j);
        }
        result.set(i, j, sum);
      }
    }
    return result;
  }

  transpose(): SimpleMatrix {
    const result = SimpleMatrix.zeros(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.set(j, i, this.get(i, j));
      }
    }
    return result;
  }

  map(fn: (value: number) => number): SimpleMatrix {
    const result = SimpleMatrix.zeros(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.set(i, j, fn(this.get(i, j)));
      }
    }
    return result;
  }

  to1DArray(): number[] {
    return this.data.flat();
  }

  // Simple inverse using Gauss-Jordan elimination (for small matrices only)
  inverse(): SimpleMatrix {
    if (this.rows !== this.cols) {
      throw new Error("Matrix must be square for inversion");
    }
    
    const n = this.rows;
    const augmented = SimpleMatrix.zeros(n, 2 * n);
    
    // Create augmented matrix [A | I]
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        augmented.set(i, j, this.get(i, j));
        augmented.set(i, j + n, i === j ? 1 : 0);
      }
    }
    
    // Gauss-Jordan elimination
    for (let i = 0; i < n; i++) {
      // Find pivot
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(augmented.get(k, i)) > Math.abs(augmented.get(maxRow, i))) {
          maxRow = k;
        }
      }
      
      // Swap rows
      if (maxRow !== i) {
        for (let j = 0; j < 2 * n; j++) {
          const temp = augmented.get(i, j);
          augmented.set(i, j, augmented.get(maxRow, j));
          augmented.set(maxRow, j, temp);
        }
      }
      
      // Check for singular matrix
      if (Math.abs(augmented.get(i, i)) < 1e-10) {
        throw new Error("Matrix is singular and cannot be inverted");
      }
      
      // Scale pivot row
      const pivot = augmented.get(i, i);
      for (let j = 0; j < 2 * n; j++) {
        augmented.set(i, j, augmented.get(i, j) / pivot);
      }
      
      // Eliminate column
      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = augmented.get(k, i);
          for (let j = 0; j < 2 * n; j++) {
            augmented.set(k, j, augmented.get(k, j) - factor * augmented.get(i, j));
          }
        }
      }
    }
    
    // Extract inverse matrix
    const inverse = SimpleMatrix.zeros(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        inverse.set(i, j, augmented.get(i, j + n));
      }
    }
    
    return inverse;
  }
}

export interface DeepTreeESNConfig {
  // Tree structure
  depth: number;
  branchingFactor: number;
  
  // Network parameters per layer
  inputSize: number;
  reservoirSizes: number[]; // Size for each layer
  outputSize: number;
  
  // Echo state network parameters
  spectralRadius?: number;
  connectivity?: number;
  inputScaling?: number;
  biasScaling?: number;
  leakingRate?: number;
  
  // Inter-layer connections
  interLayerConnectivity?: number;
  hierarchicalScaling?: number;
}

interface ESNLayer {
  id: string;
  level: number;
  parentId?: string;
  childIds: string[];
  
  // ESN components
  inputWeights: SimpleMatrix;
  reservoirWeights: SimpleMatrix;
  biasWeights: SimpleMatrix;
  outputWeights?: SimpleMatrix;
  reservoirState: SimpleMatrix;
  
  // Layer configuration
  inputSize: number;
  reservoirSize: number;
  outputSize: number;
}

export class DeepTreeEchoStateNetwork {
  private config: DeepTreeESNConfig;
  private layers: Map<string, ESNLayer>;
  private rootLayerId: string;
  private leafLayerIds: string[];
  private trained: boolean = false;

  constructor(config: DeepTreeESNConfig) {
    this.config = {
      spectralRadius: 0.99,
      connectivity: 0.1,
      inputScaling: 1.0,
      biasScaling: 0.1,
      leakingRate: 1.0,
      interLayerConnectivity: 0.05,
      hierarchicalScaling: 0.8,
      ...config
    };
    
    this.layers = new Map();
    this.rootLayerId = '';
    this.leafLayerIds = [];
    
    this.initializeTreeStructure();
  }

  private initializeTreeStructure(): void {
    const { depth, branchingFactor, inputSize, reservoirSizes, outputSize } = this.config;
    
    // Create tree structure
    const queue: Array<{ id: string; level: number; parentId?: string }> = [];
    
    // Root layer
    this.rootLayerId = 'root';
    queue.push({ id: this.rootLayerId, level: 0 });
    
    while (queue.length > 0) {
      const { id, level, parentId } = queue.shift()!;
      
      // Determine layer sizes
      const layerInputSize = level === 0 ? inputSize : reservoirSizes[level - 1] || reservoirSizes[reservoirSizes.length - 1];
      const layerReservoirSize = reservoirSizes[level] || reservoirSizes[reservoirSizes.length - 1];
      const layerOutputSize = level === depth - 1 ? outputSize : layerReservoirSize;
      
      // Create layer
      const layer: ESNLayer = {
        id,
        level,
        parentId,
        childIds: [],
        inputSize: layerInputSize,
        reservoirSize: layerReservoirSize,
        outputSize: layerOutputSize,
        inputWeights: this.initializeInputWeights(layerReservoirSize, layerInputSize),
        reservoirWeights: this.initializeReservoirWeights(layerReservoirSize),
        biasWeights: this.initializeBiasWeights(layerReservoirSize),
        reservoirState: SimpleMatrix.zeros(1, layerReservoirSize)
      };
      
      this.layers.set(id, layer);
      
      // Add to parent's children
      if (parentId) {
        const parent = this.layers.get(parentId);
        if (parent) {
          parent.childIds.push(id);
        }
      }
      
      // Create children if not at max depth
      if (level < depth - 1) {
        for (let i = 0; i < branchingFactor; i++) {
          const childId = `${id}_${i}`;
          queue.push({ id: childId, level: level + 1, parentId: id });
        }
      } else {
        // This is a leaf layer
        this.leafLayerIds.push(id);
      }
    }
  }

  private initializeInputWeights(reservoirSize: number, inputSize: number): SimpleMatrix {
    const weights = SimpleMatrix.random(reservoirSize, inputSize);
    return weights.mul(this.config.inputScaling!);
  }

  private initializeReservoirWeights(reservoirSize: number): SimpleMatrix {
    // Create sparse random matrix
    const weights = SimpleMatrix.zeros(reservoirSize, reservoirSize);
    const totalElements = reservoirSize * reservoirSize;
    const nonZeroElements = Math.floor(totalElements * this.config.connectivity!);

    for (let i = 0; i < nonZeroElements; i++) {
      const row = Math.floor(Math.random() * reservoirSize);
      const col = Math.floor(Math.random() * reservoirSize);
      weights.set(row, col, Math.random() * 2 - 1);
    }

    // Scale to desired spectral radius (simplified)
    return weights.mul(this.config.spectralRadius!);
  }

  private initializeBiasWeights(reservoirSize: number): SimpleMatrix {
    const weights = SimpleMatrix.random(reservoirSize, 1);
    return weights.mul(this.config.biasScaling!);
  }

  private activateLayer(layerId: string, input: number[]): void {
    const layer = this.layers.get(layerId);
    if (!layer) return;

    const inputMatrix = SimpleMatrix.columnVector(input);

    // Update reservoir state
    const inputContribution = layer.inputWeights.mmul(inputMatrix);
    const reservoirContribution = layer.reservoirWeights.mmul(
      layer.reservoirState.transpose()
    );
    const biasContribution = layer.biasWeights;

    const newState = inputContribution
      .add(reservoirContribution)
      .add(biasContribution)
      .map(Math.tanh);

    // Apply leaking rate
    layer.reservoirState = layer.reservoirState
      .mul(1 - this.config.leakingRate!)
      .add(newState.transpose().mul(this.config.leakingRate!));
  }

  public process(input: number[]): Map<string, number[]> {
    const results = new Map<string, number[]>();
    
    // Process tree from root to leaves
    this.processTreeRecursive(this.rootLayerId, input, results);
    
    return results;
  }

  private processTreeRecursive(layerId: string, input: number[], results: Map<string, number[]>): void {
    const layer = this.layers.get(layerId);
    if (!layer) return;

    // Activate current layer
    this.activateLayer(layerId, input);
    
    // Store layer output
    const layerOutput = layer.reservoirState.to1DArray();
    results.set(layerId, layerOutput);

    // Process children with current layer's output as input
    if (layer.childIds.length > 0) {
      for (const childId of layer.childIds) {
        // Apply hierarchical scaling to the output before passing to children
        const scaledOutput = layerOutput.map(x => x * this.config.hierarchicalScaling!);
        this.processTreeRecursive(childId, scaledOutput, results);
      }
    }
  }

  public train(inputs: number[][], outputs: number[][]): void {
    const allStates: number[][] = [];
    const allOutputs: number[][] = [];

    // Collect states from all training samples
    for (let i = 0; i < inputs.length; i++) {
      const results = this.process(inputs[i]);
      
      // Concatenate states from all leaf layers
      const concatenatedState: number[] = [];
      for (const leafId of this.leafLayerIds) {
        const leafState = results.get(leafId) || [];
        concatenatedState.push(...leafState);
      }
      
      allStates.push(concatenatedState);
      allOutputs.push(outputs[i]);
    }

    // Train output weights using ridge regression
    if (allStates.length > 0 && allStates[0].length > 0) {
      const X = new SimpleMatrix(allStates);
      const Y = new SimpleMatrix(allOutputs);
      const ridge = 1e-6;

      try {
        const XtX = X.transpose().mmul(X);
        const I = SimpleMatrix.eye(allStates[0].length).mul(ridge);
        const Xt = X.transpose();

        const outputWeights = XtX.add(I).inverse().mmul(Xt).mmul(Y);
        
        // Store output weights in leaf layers (distributed)
        const weightsPerLeaf = Math.ceil(outputWeights.rows / this.leafLayerIds.length);
        let weightIndex = 0;
        
        for (const leafId of this.leafLayerIds) {
          const layer = this.layers.get(leafId);
          if (layer) {
            const startRow = weightIndex;
            const endRow = Math.min(startRow + weightsPerLeaf, outputWeights.rows);
            
            // Extract portion of weights for this leaf
            const leafWeights = SimpleMatrix.zeros(endRow - startRow, outputWeights.cols);
            for (let i = startRow; i < endRow; i++) {
              for (let j = 0; j < outputWeights.cols; j++) {
                leafWeights.set(i - startRow, j, outputWeights.get(i, j));
              }
            }
            
            layer.outputWeights = leafWeights;
            weightIndex = endRow;
          }
        }
        
        this.trained = true;
      } catch (error) {
        console.warn('Training failed, using random output weights:', error);
        // Fallback to random weights
        for (const leafId of this.leafLayerIds) {
          const layer = this.layers.get(leafId);
          if (layer) {
            layer.outputWeights = SimpleMatrix.random(layer.reservoirSize, this.config.outputSize);
          }
        }
        this.trained = true;
      }
    }
  }

  public predict(input: number[]): number[] {
    if (!this.trained) {
      throw new Error("Network not trained");
    }

    const results = this.process(input);
    const finalOutput: number[] = [];

    // Combine outputs from all leaf layers
    for (const leafId of this.leafLayerIds) {
      const layer = this.layers.get(leafId);
      const leafState = results.get(leafId);
      
      if (layer && layer.outputWeights && leafState) {
        const stateMatrix = SimpleMatrix.rowVector(leafState);
        const output = stateMatrix.mmul(layer.outputWeights);
        finalOutput.push(...output.to1DArray());
      }
    }

    return finalOutput.slice(0, this.config.outputSize);
  }

  public reset(): void {
    for (const layer of this.layers.values()) {
      layer.reservoirState = SimpleMatrix.zeros(1, layer.reservoirSize);
    }
  }

  public getTreeStructure(): object {
    const buildTree = (layerId: string): any => {
      const layer = this.layers.get(layerId);
      if (!layer) return null;

      return {
        id: layer.id,
        level: layer.level,
        reservoirSize: layer.reservoirSize,
        children: layer.childIds.map(childId => buildTree(childId))
      };
    };

    return buildTree(this.rootLayerId);
  }

  public getLayerStates(): Map<string, number[]> {
    const states = new Map<string, number[]>();
    for (const [layerId, layer] of this.layers) {
      states.set(layerId, layer.reservoirState.to1DArray());
    }
    return states;
  }

  public getConfig(): DeepTreeESNConfig {
    return { ...this.config };
  }
}