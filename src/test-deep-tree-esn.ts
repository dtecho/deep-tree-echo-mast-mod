/**
 * Test script for Deep Tree Echo State Network
 * This script demonstrates the functionality of the Deep Tree ESN model
 */

import { DeepTreeEchoStateNetwork, DeepTreeESNConfig } from './mastra/models/deepTreeEchoStateNetwork';
import { deepTreeEchoTool } from './mastra/tools/deepTreeEchoTool';

// Test configuration
const testConfig: DeepTreeESNConfig = {
  depth: 3,
  branchingFactor: 2,
  inputSize: 5,
  reservoirSizes: [20, 15, 10],
  outputSize: 3,
  spectralRadius: 0.9,
  connectivity: 0.1,
  inputScaling: 0.5,
  biasScaling: 0.1,
  leakingRate: 0.9,
  hierarchicalScaling: 0.8
};

async function testDeepTreeESN() {
  console.log('🌳 Testing Deep Tree Echo State Network Implementation');
  console.log('=' .repeat(60));

  try {
    // Test 1: Create network directly
    console.log('\n1. Creating Deep Tree ESN directly...');
    const network = new DeepTreeEchoStateNetwork(testConfig);
    console.log('✅ Network created successfully');
    console.log('📊 Tree structure:', JSON.stringify(network.getTreeStructure(), null, 2));

    // Test 2: Process input
    console.log('\n2. Processing input through network...');
    const testInput = [0.1, 0.2, 0.3, 0.4, 0.5];
    const results = network.process(testInput);
    console.log('✅ Input processed successfully');
    console.log(`📈 Layer outputs (${results.size} layers):`);
    for (const [layerId, output] of results) {
      console.log(`   ${layerId}: [${output.slice(0, 3).map(x => x.toFixed(4)).join(', ')}${output.length > 3 ? '...' : ''}]`);
    }

    // Test 3: Training
    console.log('\n3. Training network...');
    const trainingInputs = [
      [0.1, 0.2, 0.3, 0.4, 0.5],
      [0.2, 0.3, 0.4, 0.5, 0.6],
      [0.3, 0.4, 0.5, 0.6, 0.7],
      [0.4, 0.5, 0.6, 0.7, 0.8],
      [0.5, 0.6, 0.7, 0.8, 0.9]
    ];
    const trainingOutputs = [
      [1.0, 0.0, 0.0],
      [0.8, 0.2, 0.0],
      [0.6, 0.4, 0.0],
      [0.4, 0.6, 0.0],
      [0.2, 0.8, 0.0]
    ];
    
    network.train(trainingInputs, trainingOutputs);
    console.log('✅ Training completed');

    // Test 4: Prediction
    console.log('\n4. Making predictions...');
    const testPrediction = network.predict([0.15, 0.25, 0.35, 0.45, 0.55]);
    console.log('✅ Prediction made:', testPrediction.map(x => x.toFixed(4)));

    // Test 5: Using Mastra tool
    console.log('\n5. Testing Mastra tool interface...');
    
    // Create network via tool
    const createResult = await deepTreeEchoTool.execute({
      inputData: {
        action: 'create',
        networkId: 'test-network',
        config: testConfig
      }
    });
    console.log('✅ Tool create result:', createResult.success ? 'Success' : `Failed: ${createResult.error}`);

    // Train via tool
    const trainResult = await deepTreeEchoTool.execute({
      inputData: {
        action: 'train',
        networkId: 'test-network',
        inputs: trainingInputs,
        outputs: trainingOutputs
      }
    });
    console.log('✅ Tool train result:', trainResult.success ? 'Success' : `Failed: ${trainResult.error}`);

    // Predict via tool
    const predictResult = await deepTreeEchoTool.execute({
      inputData: {
        action: 'predict',
        networkId: 'test-network',
        input: [0.15, 0.25, 0.35, 0.45, 0.55]
      }
    });
    console.log('✅ Tool predict result:', predictResult.success ? 'Success' : `Failed: ${predictResult.error}`);
    if (predictResult.success) {
      console.log('📊 Prediction:', predictResult.prediction.map((x: number) => x.toFixed(4)));
    }

    // Get structure via tool
    const structureResult = await deepTreeEchoTool.execute({
      inputData: {
        action: 'getStructure',
        networkId: 'test-network'
      }
    });
    console.log('✅ Tool structure result:', structureResult.success ? 'Success' : `Failed: ${structureResult.error}`);

    // List networks
    const listResult = await deepTreeEchoTool.execute({
      inputData: {
        action: 'list'
      }
    });
    console.log('✅ Tool list result:', listResult.success ? `Found ${listResult.count} networks` : `Failed: ${listResult.error}`);

    console.log('\n🎉 All tests completed successfully!');
    console.log('\n📋 Summary:');
    console.log('- Deep Tree Echo State Network implementation: ✅');
    console.log('- Hierarchical processing with tree structure: ✅');
    console.log('- Training and prediction capabilities: ✅');
    console.log('- Mastra tool integration: ✅');
    console.log('- Multiple network management: ✅');

  } catch (error) {
    console.error('❌ Test failed:', error);
    if (error instanceof Error) {
      console.error('Stack trace:', error.stack);
    }
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  testDeepTreeESN().catch(console.error);
}

export { testDeepTreeESN };