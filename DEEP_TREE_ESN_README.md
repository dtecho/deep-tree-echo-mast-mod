# Deep Tree Echo State Network for Mastra

## Overview

This implementation provides a **Deep Tree Echo State Network** (Deep Tree ESN) model integrated into the Mastra framework. The Deep Tree ESN extends traditional Echo State Networks by introducing a hierarchical, tree-like structure that processes information through multiple layers, creating rich, interconnected patterns of understanding.

## Features

### 🌳 Hierarchical Architecture
- **Tree Structure**: Multi-level processing with configurable depth and branching factors
- **Layer-wise Processing**: Information flows from root to leaf nodes with hierarchical scaling
- **Distributed Computation**: Each layer maintains its own reservoir state and dynamics

### 🧠 Advanced Neural Dynamics
- **Echo State Properties**: Implements proper echo state network dynamics with reservoir computing
- **Temporal Processing**: Maintains memory through recurrent connections and leaking rates
- **Pattern Recognition**: Hierarchical feature extraction and pattern identification

### 🔧 Mastra Integration
- **Tool Interface**: Complete Mastra tool for network management
- **Agent Integration**: Enhanced Deep Tree Echo Agent with ESN capabilities
- **State Management**: Persistent network instances with configuration management

## Architecture

### Core Components

```
Root Layer (Level 0)
├── Branch 1 (Level 1)
│   ├── Leaf 1.1 (Level 2)
│   └── Leaf 1.2 (Level 2)
└── Branch 2 (Level 1)
    ├── Leaf 2.1 (Level 2)
    └── Leaf 2.2 (Level 2)
```

### Layer Structure
Each layer contains:
- **Input Weights**: Transform input signals
- **Reservoir Weights**: Recurrent connections within the layer
- **Bias Weights**: Bias terms for activation
- **Reservoir State**: Current activation state
- **Output Weights**: (Leaf layers only) Transform reservoir states to outputs

## Configuration

### DeepTreeESNConfig Interface

```typescript
interface DeepTreeESNConfig {
  // Tree structure
  depth: number;                    // Number of levels (1-10)
  branchingFactor: number;          // Children per node (1-5)
  
  // Network parameters
  inputSize: number;                // Input vector dimension
  reservoirSizes: number[];         // Reservoir size per level
  outputSize: number;               // Output vector dimension
  
  // ESN parameters (optional)
  spectralRadius?: number;          // Max eigenvalue magnitude (default: 0.99)
  connectivity?: number;            // Sparsity of connections (default: 0.1)
  inputScaling?: number;            // Input weight scaling (default: 1.0)
  biasScaling?: number;             // Bias weight scaling (default: 0.1)
  leakingRate?: number;             // State update rate (default: 1.0)
  
  // Hierarchical parameters
  interLayerConnectivity?: number;  // Inter-layer connection density (default: 0.05)
  hierarchicalScaling?: number;     // Output scaling between levels (default: 0.8)
}
```

### Example Configuration

```typescript
const config: DeepTreeESNConfig = {
  depth: 3,
  branchingFactor: 2,
  inputSize: 10,
  reservoirSizes: [50, 30, 20],
  outputSize: 5,
  spectralRadius: 0.9,
  connectivity: 0.1,
  leakingRate: 0.9,
  hierarchicalScaling: 0.8
};
```

## Usage

### Direct Usage

```typescript
import { DeepTreeEchoStateNetwork } from './src/mastra/models/deepTreeEchoStateNetwork';

// Create network
const network = new DeepTreeEchoStateNetwork(config);

// Process input
const input = [0.1, 0.2, 0.3, 0.4, 0.5];
const results = network.process(input);

// Train network
const trainingInputs = [[...], [...], ...];
const trainingOutputs = [[...], [...], ...];
network.train(trainingInputs, trainingOutputs);

// Make predictions
const prediction = network.predict(input);
```

### Mastra Tool Usage

```typescript
import { deepTreeEchoTool } from './src/mastra/tools/deepTreeEchoTool';

// Create network
const createResult = await deepTreeEchoTool.execute({
  inputData: {
    action: 'create',
    networkId: 'my-network',
    config: config
  }
});

// Train network
const trainResult = await deepTreeEchoTool.execute({
  inputData: {
    action: 'train',
    networkId: 'my-network',
    inputs: trainingInputs,
    outputs: trainingOutputs
  }
});

// Make prediction
const predictResult = await deepTreeEchoTool.execute({
  inputData: {
    action: 'predict',
    networkId: 'my-network',
    input: input
  }
});
```

### Agent Integration

The Deep Tree Echo Agent automatically includes the Deep Tree ESN tool:

```typescript
import { deepTreeEchoAgent } from './src/mastra/agents/deepTreeEchoAgent';

// The agent can now use Deep Tree ESN capabilities
// through natural language conversations
```

## Tool Actions

### Available Actions

1. **create**: Create a new Deep Tree ESN
2. **train**: Train the network with input/output pairs
3. **predict**: Generate predictions from input
4. **process**: Process input and get all layer outputs
5. **reset**: Reset network state to zero
6. **getStructure**: Retrieve network architecture
7. **getStates**: Get current reservoir states
8. **list**: List all managed networks

### Action Examples

```typescript
// Create with custom configuration
{
  action: 'create',
  networkId: 'conversation-processor',
  config: {
    depth: 4,
    branchingFactor: 3,
    inputSize: 1536,  // OpenAI embedding size
    reservoirSizes: [512, 256, 128, 64],
    outputSize: 256
  }
}

// Train on conversation data
{
  action: 'train',
  networkId: 'conversation-processor',
  inputs: embeddings,
  outputs: responses
}

// Process real-time input
{
  action: 'process',
  networkId: 'conversation-processor',
  input: currentEmbedding
}
```

## Mathematical Foundation

### Reservoir Dynamics

For each layer `l` at time `t`:

```
x_l(t) = (1 - α) * x_l(t-1) + α * tanh(W_in * u_l(t) + W_res * x_l(t-1) + W_bias)
```

Where:
- `x_l(t)`: Reservoir state at layer `l` and time `t`
- `u_l(t)`: Input to layer `l` at time `t`
- `α`: Leaking rate
- `W_in`: Input weights
- `W_res`: Reservoir weights
- `W_bias`: Bias weights

### Inter-layer Information Flow

```
u_l(t) = hierarchicalScaling * x_{parent(l)}(t)  for l > 0
```

### Output Generation

```
y(t) = W_out * [x_leaf1(t), x_leaf2(t), ..., x_leafN(t)]
```

## Performance Characteristics

### Computational Complexity
- **Training**: O(N³) where N is total reservoir size (due to matrix inversion)
- **Prediction**: O(N²) where N is total reservoir size
- **Memory**: O(N²) for weight storage

### Scalability
- Depth: Recommended 2-5 layers for most applications
- Branching Factor: 2-4 for balanced computation/capacity trade-off
- Reservoir Size: 50-500 neurons per layer depending on complexity

## Use Cases

### 1. Conversation Processing
- Process embeddings of conversation turns
- Maintain hierarchical context across multiple levels
- Generate contextually aware responses

### 2. Time Series Analysis
- Multi-scale temporal pattern recognition
- Hierarchical feature extraction
- Long-term dependency modeling

### 3. Pattern Recognition
- Complex sequential pattern identification
- Multi-level feature hierarchy
- Adaptive pattern learning

### 4. Memory Processing
- Hierarchical memory organization
- Multi-timescale memory dynamics
- Associative memory networks

## Testing

The implementation includes comprehensive tests:

```bash
# Test core functionality
npx tsx src/test-core-functionality.ts

# Test tool functionality
npx tsx src/test-tool-functionality.ts
```

### Test Coverage
- ✅ Network creation and initialization
- ✅ Hierarchical processing
- ✅ Training with ridge regression
- ✅ Prediction generation
- ✅ State management
- ✅ Tool interface functionality
- ✅ Multiple network management

## Advanced Features

### Custom Matrix Operations
- Self-contained matrix library (no external dependencies)
- Optimized for small to medium-sized matrices
- Includes necessary operations: multiplication, inversion, transpose

### Error Handling
- Graceful degradation with fallback weights
- Comprehensive error messages
- Network state validation

### State Persistence
- Network state can be retrieved and restored
- Configuration persistence
- Multi-network management

## Future Enhancements

### Potential Improvements
1. **GPU Acceleration**: CUDA/WebGL implementations
2. **Online Learning**: Incremental training capabilities
3. **Attention Mechanisms**: Inter-layer attention weights
4. **Compression**: Network pruning and quantization
5. **Visualization**: Real-time network state visualization

### Research Directions
1. **Adaptive Topology**: Dynamic tree structure modification
2. **Meta-Learning**: Network architecture optimization
3. **Federated Learning**: Distributed Deep Tree ESN training
4. **Neuroevolution**: Evolutionary architecture search

## References

1. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks
2. Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training
3. Gallicchio, C., & Micheli, A. (2017). Deep reservoir computing: A critical experimental analysis

## License

This implementation is part of the Deep Tree Echo Mast Mod project and follows the project's licensing terms.

---

*Built with 🌳 for the Mastra ecosystem*