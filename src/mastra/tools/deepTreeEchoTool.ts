import { DeepTreeEchoStateNetwork, DeepTreeESNConfig } from '../models/deepTreeEchoStateNetwork';

// Global instances to maintain state
const deepTreeNetworks = new Map<string, DeepTreeEchoStateNetwork>();

// Simple validation function to replace zod
function validateConfig(config: any): DeepTreeESNConfig {
  const validated: DeepTreeESNConfig = {
    depth: Math.max(1, Math.min(10, config.depth || 3)),
    branchingFactor: Math.max(1, Math.min(5, config.branchingFactor || 2)),
    inputSize: Math.max(1, config.inputSize || 10),
    reservoirSizes: Array.isArray(config.reservoirSizes) ? config.reservoirSizes : [50, 30, 20],
    outputSize: Math.max(1, config.outputSize || 10),
    spectralRadius: config.spectralRadius !== undefined ? Math.max(0, Math.min(2, config.spectralRadius)) : 0.99,
    connectivity: config.connectivity !== undefined ? Math.max(0, Math.min(1, config.connectivity)) : 0.1,
    inputScaling: config.inputScaling !== undefined ? Math.max(0, config.inputScaling) : 1.0,
    biasScaling: config.biasScaling !== undefined ? Math.max(0, config.biasScaling) : 0.1,
    leakingRate: config.leakingRate !== undefined ? Math.max(0, Math.min(1, config.leakingRate)) : 1.0,
    interLayerConnectivity: config.interLayerConnectivity !== undefined ? Math.max(0, Math.min(1, config.interLayerConnectivity)) : 0.05,
    hierarchicalScaling: config.hierarchicalScaling !== undefined ? Math.max(0, Math.min(1, config.hierarchicalScaling)) : 0.8,
  };
  
  return validated;
}

export const deepTreeEchoTool = {
  id: 'deep_tree_echo_tool',
  description: 'A tool for managing Deep Tree Echo State Networks - hierarchical neural networks for complex pattern recognition and temporal processing',
  inputSchema: {
    type: 'object',
    properties: {
      action: {
        type: 'string',
        enum: ['create', 'train', 'predict', 'process', 'reset', 'getStructure', 'getStates', 'list'],
        description: 'Action to perform on the Deep Tree ESN'
      },
      networkId: {
        type: 'string',
        default: 'default',
        description: 'Unique identifier for the network'
      },
      config: {
        type: 'object',
        description: 'Configuration for creating a new network'
      },
      inputs: {
        type: 'array',
        description: 'Training input data (array of arrays)'
      },
      outputs: {
        type: 'array',
        description: 'Training output data (array of arrays)'
      },
      input: {
        type: 'array',
        description: 'Single input vector for prediction or processing'
      }
    },
    required: ['action']
  },
  execute: async ({ inputData }: { inputData: any }) => {
    const { action, networkId, config, inputs, outputs, input } = inputData;

    try {
      switch (action) {
        case 'create': {
          if (!config) {
            throw new Error('Configuration required for creating network');
          }
          
          const validatedConfig = validateConfig(config);
          const network = new DeepTreeEchoStateNetwork(validatedConfig);
          deepTreeNetworks.set(networkId, network);
          
          return {
            success: true,
            message: `Deep Tree Echo State Network '${networkId}' created successfully`,
            networkId,
            config: validatedConfig,
            structure: network.getTreeStructure(),
          };
        }

        case 'train': {
          const network = deepTreeNetworks.get(networkId);
          if (!network) {
            throw new Error(`Network '${networkId}' not found. Create it first.`);
          }
          
          if (!inputs || !outputs) {
            throw new Error('Training inputs and outputs are required');
          }
          
          if (inputs.length !== outputs.length) {
            throw new Error('Number of inputs must match number of outputs');
          }
          
          network.train(inputs, outputs);
          
          return {
            success: true,
            message: `Network '${networkId}' trained with ${inputs.length} samples`,
            networkId,
            trainingSamples: inputs.length,
          };
        }

        case 'predict': {
          const network = deepTreeNetworks.get(networkId);
          if (!network) {
            throw new Error(`Network '${networkId}' not found. Create it first.`);
          }
          
          if (!input) {
            throw new Error('Input vector is required for prediction');
          }
          
          const prediction = network.predict(input);
          
          return {
            success: true,
            networkId,
            input,
            prediction,
            predictionLength: prediction.length,
          };
        }

        case 'process': {
          const network = deepTreeNetworks.get(networkId);
          if (!network) {
            throw new Error(`Network '${networkId}' not found. Create it first.`);
          }
          
          if (!input) {
            throw new Error('Input vector is required for processing');
          }
          
          const results = network.process(input);
          const processedResults: Record<string, number[]> = {};
          
          for (const [layerId, output] of results) {
            processedResults[layerId] = output;
          }
          
          return {
            success: true,
            networkId,
            input,
            layerOutputs: processedResults,
            layerCount: Object.keys(processedResults).length,
          };
        }

        case 'reset': {
          const network = deepTreeNetworks.get(networkId);
          if (!network) {
            throw new Error(`Network '${networkId}' not found. Create it first.`);
          }
          
          network.reset();
          
          return {
            success: true,
            message: `Network '${networkId}' state reset successfully`,
            networkId,
          };
        }

        case 'getStructure': {
          const network = deepTreeNetworks.get(networkId);
          if (!network) {
            throw new Error(`Network '${networkId}' not found. Create it first.`);
          }
          
          const structure = network.getTreeStructure();
          const config = network.getConfig();
          
          return {
            success: true,
            networkId,
            structure,
            config,
          };
        }

        case 'getStates': {
          const network = deepTreeNetworks.get(networkId);
          if (!network) {
            throw new Error(`Network '${networkId}' not found. Create it first.`);
          }
          
          const states = network.getLayerStates();
          const stateRecord: Record<string, number[]> = {};
          
          for (const [layerId, state] of states) {
            stateRecord[layerId] = state;
          }
          
          return {
            success: true,
            networkId,
            layerStates: stateRecord,
            layerCount: Object.keys(stateRecord).length,
          };
        }

        case 'list': {
          const networkList = Array.from(deepTreeNetworks.keys()).map(id => {
            const network = deepTreeNetworks.get(id);
            return {
              id,
              config: network?.getConfig(),
              structure: network?.getTreeStructure(),
            };
          });
          
          return {
            success: true,
            networks: networkList,
            count: networkList.length,
          };
        }

        default:
          throw new Error(`Unknown action: ${action}`);
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
        networkId,
        action,
      };
    }
  },
};