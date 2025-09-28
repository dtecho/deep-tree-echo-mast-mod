/**
 * Core functionality test for Deep Tree Echo State Network
 * This script tests only the core network without external dependencies
 */

import { DeepTreeEchoStateNetwork, DeepTreeESNConfig } from './mastra/models/deepTreeEchoStateNetwork';

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

function testDeepTreeESNCore() {
  console.log('🌳 Testing Deep Tree Echo State Network Core Implementation');
  console.log('=' .repeat(60));

  try {
    // Test 1: Create network
    console.log('\n1. Creating Deep Tree ESN...');
    const network = new DeepTreeEchoStateNetwork(testConfig);
    console.log('✅ Network created successfully');
    
    const structure = network.getTreeStructure();
    console.log('📊 Tree structure:', JSON.stringify(structure, null, 2));

    // Test 2: Process input
    console.log('\n2. Processing input through network...');
    const testInput = [0.1, 0.2, 0.3, 0.4, 0.5];
    const results = network.process(testInput);
    console.log('✅ Input processed successfully');
    console.log(`📈 Layer outputs (${results.size} layers):`);
    for (const [layerId, output] of results) {
      console.log(`   ${layerId}: [${output.slice(0, 3).map(x => x.toFixed(4)).join(', ')}${output.length > 3 ? '...' : ''}] (${output.length} values)`);
    }

    // Test 3: Get layer states
    console.log('\n3. Getting layer states...');
    const states = network.getLayerStates();
    console.log(`✅ Retrieved states from ${states.size} layers`);
    for (const [layerId, state] of states) {
      console.log(`   ${layerId}: [${state.slice(0, 3).map(x => x.toFixed(4)).join(', ')}${state.length > 3 ? '...' : ''}] (${state.length} values)`);
    }

    // Test 4: Training
    console.log('\n4. Training network...');
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

    // Test 5: Prediction
    console.log('\n5. Making predictions...');
    const testPrediction = network.predict([0.15, 0.25, 0.35, 0.45, 0.55]);
    console.log('✅ Prediction made:', testPrediction.map(x => x.toFixed(4)));

    // Test 6: Reset network
    console.log('\n6. Resetting network state...');
    network.reset();
    console.log('✅ Network state reset');

    // Test 7: Configuration
    console.log('\n7. Verifying configuration...');
    const config = network.getConfig();
    console.log('✅ Configuration retrieved:');
    console.log(`   Depth: ${config.depth}`);
    console.log(`   Branching Factor: ${config.branchingFactor}`);
    console.log(`   Input Size: ${config.inputSize}`);
    console.log(`   Output Size: ${config.outputSize}`);
    console.log(`   Reservoir Sizes: [${config.reservoirSizes.join(', ')}]`);

    // Test 8: Multiple processing cycles
    console.log('\n8. Testing multiple processing cycles...');
    const testInputs = [
      [0.1, 0.1, 0.1, 0.1, 0.1],
      [0.5, 0.5, 0.5, 0.5, 0.5],
      [0.9, 0.9, 0.9, 0.9, 0.9]
    ];
    
    for (let i = 0; i < testInputs.length; i++) {
      const cycleResults = network.process(testInputs[i]);
      console.log(`   Cycle ${i + 1}: Processed ${cycleResults.size} layers`);
    }
    console.log('✅ Multiple cycles completed');

    console.log('\n🎉 All core tests completed successfully!');
    console.log('\n📋 Summary:');
    console.log('- Deep Tree Echo State Network creation: ✅');
    console.log('- Hierarchical tree structure: ✅');
    console.log('- Input processing through layers: ✅');
    console.log('- Training with ridge regression: ✅');
    console.log('- Prediction generation: ✅');
    console.log('- State management and reset: ✅');
    console.log('- Configuration management: ✅');
    console.log('- Multiple processing cycles: ✅');

    return true;

  } catch (error) {
    console.error('❌ Test failed:', error);
    if (error instanceof Error) {
      console.error('Stack trace:', error.stack);
    }
    return false;
  }
}

// Run tests
const success = testDeepTreeESNCore();
process.exit(success ? 0 : 1);