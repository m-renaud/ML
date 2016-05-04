module NN.Tests (tests) where

import NN (ActivationFunction, Layer(..), Network(..), feedForward, runNetwork)

import Numeric.LinearAlgebra (konst, size, sumElements, vector)

import Test.HUnit (Assertion, (@?=))
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (testCase)

identityActivation :: ActivationFunction
identityActivation x = x

tests :: TestTree
tests = testGroup "NN"
    [
        testCase
        "testFeedForward_SumInputWeights_NoBias" testFeedForward_SumInputWeights_NoBias
    ,   testCase
        "runNetwork_Empty" testRunNetwork_Empty
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
          emptyNetwork = Network []
