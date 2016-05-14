module ML.NN.Tests (tests) where

import ML.NN (ActivationFunction, Layer(..), Network(..), computeActivations,
              feedForward, runNetwork)

import Numeric.LinearAlgebra ((><), konst, size, sumElements, vector)

import Test.HUnit (Assertion, (@?=))
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (testCase)

identityActivation :: ActivationFunction
identityActivation x = x

emptyNetwork :: Network
emptyNetwork = Network []

tests :: TestTree
tests = testGroup "ML.NN"
    [
        testCase
        "FeedForward_SumInputWeights_NoBias" testFeedForward_SumInputWeights_NoBias
    ,   testCase
        "RunNetwork_Empty" testRunNetwork_Empty
    ,   testCase
        "ComputeActivations_EmptyNetwork" testComputeActivations_EmptyNetwork
    ,   testCase
        "ComputeActivations_SingleLayerSingleNeuron" testComputeActivations_SingleLayerSingleNeuron
    ]


-- | Test a simple layer where each neuron sums the input.
testFeedForward_SumInputWeights_NoBias :: Assertion
testFeedForward_SumInputWeights_NoBias =
    feedForward identityActivation x sumLayer @?= expected
    where x          = vector [1,2,3]
          expected   = vector $ replicate numNeurons (sumElements x)
          numNeurons = 5
          sumLayer   = Layer (konst 0 numNeurons) (konst 1 (numNeurons, size x))

-- | Test that running an empty network on an input 'x' yields 'x'.
testRunNetwork_Empty :: Assertion
testRunNetwork_Empty =
    runNetwork emptyNetwork identityActivation x @?= x
    where x = vector [1,2,3]

-- | When there are no layers, there should be no z's and the only
--   activation should be the input.
testComputeZsAndAs_EmptyNetwork :: Assertion
testComputeZsAndAs_EmptyNetwork =
    computeZsAndAs identityActivation emptyNetwork x @?= ([], [x])
    where x = vector [1,2,3]

-- | When there is a single identity layer, the zs = [z] where z is the sum
--   of the input values, and the activations are z and the input x.
--
--   The order of the activations is reversed from the order they appear in the
--   network so they can easily be used by the backpropogation algorithm.
testComputeZsAndAs_SingleLayerSingleNeuron :: Assertion
testComputeZsAndAs_SingleLayerSingleNeuron =
    computeZsAndAs identityActivation net x @?= ([z], [z,x])
    where x = vector [1,2,3]
          z = vector [6]
          net = Network [Layer (vector [0]) ((1><3) [1,1,1])]
