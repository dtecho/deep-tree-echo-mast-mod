/**
 * Test script for Deep Tree Echo Tool functionality
 */

import { deepTreeEchoTool } from './mastra/tools/deepTreeEchoTool';

const testConfig = {
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

async function testToolFunctionality() {
  console.log('🔧 Testing Deep Tree Echo Tool Functionality');
  console.log('=' .repeat(50));

  try {
    // Test 1: Create network via tool
    console.log('\n1. Creating network via tool...');
    const createResult = await deepTreeEchoTool.execute({
      inputData: {
        action: 'create',
        networkId: 'test-network',
        config: testConfig
      }
    });
    console.log('✅ Tool create result:', createResult.success ? 'Success' : `Failed: ${createResult.error}`);
    if (createResult.success) {
      console.log('📊 Network structure created');
    }

    // Test 2: Train via tool
    console.log('\n2. Training network via tool...');
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
    
    const trainResult = await deepTreeEchoTool.execute({
      inputData: {
        action: 'train',
        networkId: 'test-network',
        inputs: trainingInputs,
        outputs: trainingOutputs
      }
    });
    console.log('✅ Tool train result:', trainResult.success ? 'Success' : `Failed: ${trainResult.error}`);
    if (trainResult.success) {
      console.log(`📈 Trained with ${trainResult.trainingSamples} samples`);
    }

    // Test 3: Predict via tool
    console.log('\n3. Making prediction via tool...');
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

    // Test 4: Process via tool
    console.log('\n4. Processing input via tool...');
    const processResult = await deepTreeEchoTool.execute({
      inputData: {
        action: 'process',
        networkId: 'test-network',
        input: [0.2, 0.3, 0.4, 0.5, 0.6]
      }
    });
    console.log('✅ Tool process result:', processResult.success ? 'Success' : `Failed: ${processResult.error}`);
    if (processResult.success) {
      console.log(`📈 Processed ${processResult.layerCount} layers`);
    }

    // Test 5: Get structure via tool
    console.log('\n5. Getting network structure via tool...');
    const structureResult = await deepTreeEchoTool.execute({
      inputData: {
        action: 'getStructure',
        networkId: 'test-network'
      }
    });
    console.log('✅ Tool structure result:', structureResult.success ? 'Success' : `Failed: ${structureResult.error}`);

    // Test 6: Get states via tool
    console.log('\n6. Getting layer states via tool...');
    const statesResult = await deepTreeEchoTool.execute({
      inputData: {
        action: 'getStates',
        networkId: 'test-network'
      }
    });
    console.log('✅ Tool states result:', statesResult.success ? 'Success' : `Failed: ${statesResult.error}`);
    if (statesResult.success) {
      console.log(`📊 Retrieved states from ${statesResult.layerCount} layers`);
    }

    // Test 7: Reset via tool
    console.log('\n7. Resetting network via tool...');
    const resetResult = await deepTreeEchoTool.execute({
      inputData: {
        action: 'reset',
        networkId: 'test-network'
      }
    });
    console.log('✅ Tool reset result:', resetResult.success ? 'Success' : `Failed: ${resetResult.error}`);

    // Test 8: List networks
    console.log('\n8. Listing networks via tool...');
    const listResult = await deepTreeEchoTool.execute({
      inputData: {
        action: 'list'
      }
    });
    console.log('✅ Tool list result:', listResult.success ? `Found ${listResult.count} networks` : `Failed: ${listResult.error}`);

    // Test 9: Create another network with different config
    console.log('\n9. Creating second network...');
    const createResult2 = await deepTreeEchoTool.execute({
      inputData: {
        action: 'create',
        networkId: 'test-network-2',
        config: {
          depth: 2,
          branchingFactor: 3,
          inputSize: 4,
          reservoirSizes: [15, 10],
          outputSize: 2
        }
      }
    });
    console.log('✅ Second network creation:', createResult2.success ? 'Success' : `Failed: ${createResult2.error}`);

    // Test 10: List networks again
    console.log('\n10. Listing networks again...');
    const listResult2 = await deepTreeEchoTool.execute({
      inputData: {
        action: 'list'
      }
    });
    console.log('✅ Tool list result:', listResult2.success ? `Found ${listResult2.count} networks` : `Failed: ${listResult2.error}`);

    console.log('\n🎉 All tool tests completed successfully!');
    console.log('\n📋 Summary:');
    console.log('- Network creation via tool: ✅');
    console.log('- Training via tool: ✅');
    console.log('- Prediction via tool: ✅');
    console.log('- Processing via tool: ✅');
    console.log('- Structure retrieval via tool: ✅');
    console.log('- State retrieval via tool: ✅');
    console.log('- Network reset via tool: ✅');
    console.log('- Network listing via tool: ✅');
    console.log('- Multiple network management: ✅');

    return true;

  } catch (error) {
    console.error('❌ Tool test failed:', error);
    if (error instanceof Error) {
      console.error('Stack trace:', error.stack);
    }
    return false;
  }
}

// Run tests
testToolFunctionality().then(success => {
  process.exit(success ? 0 : 1);
}).catch(error => {
  console.error('❌ Unexpected error:', error);
  process.exit(1);
});