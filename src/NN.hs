{-# LANGUAGE BangPatterns #-}
module NN
       (
         -- * Data types
         ActivationFunction
       , Neuron
       , Layer
       , Network

         -- * Network initialization
       , randNeuron
       , randLayer
       , randNetwork

         -- * Running networks
       , feedForward
       , runLayer
       , runNetwork

         -- * Activation functions
       , sigmoid
       ) where

import Control.Monad (forM, replicateM)
import Control.Monad.Random (Rand, liftRand)
import Data.Random.Normal (normal)
import Numeric.LinearAlgebra (R, Vector, dot, vector)
import System.Random (RandomGen)

-- ==================================================
-- Types.
-- ==================================================

-- | The activation function maps R -> R.
type ActivationFunction = R -> R

-- | A Neuron has a bias and weights for each of its inputs.
data Neuron = Neuron !R (Vector R) deriving Show

-- | A layer in a neural network is a list of Neurons with weights
--  for the outputs of the neurons from the previous level.
data Layer = Layer [Neuron] deriving Show

-- | A network is a list of layers.
data Network = Network [Layer] deriving Show


-- ==================================================
-- Network initialization.
-- ==================================================

-- | Generate a random neuron given the number of inputs.
--
-- The bias and weights are taken from a normal distribution with
-- mean 0 and standard deviation of 1.
randNeuron :: RandomGen g => Int -> Rand g Neuron
randNeuron numInputs = do
    bias    <- liftRand normal
    weights <- vector <$> replicateM numInputs (liftRand normal)
    return $ Neuron bias weights

-- | Generate a random layer in the neural network given the number of inputs and neurons.
randLayer :: RandomGen g => Int -> Int -> Rand g Layer
randLayer numInputs numNeurons =
    Layer <$> replicateM numNeurons (randNeuron numInputs)


-- | Generate a random neural network given the sizes of each of the layers.
--
-- 'sizes' is a list of the number of neurons in each layer. For example, [3,2,4]
-- will generate a neural network with 3 input neurons, 1 hidden layer with 2 neurons
-- and 4 output neurons.
randNetwork :: RandomGen g => [Int] -> Rand g Network
randNetwork sizes = Network <$> forM dims (uncurry randLayer)
    where dims = adjacentPairs sizes

-- | Return a list of pairs of adjacent elements in a list.
--
-- Example: adjacentPairs [1,2,3] ==> [(1,2), (2,3)]
adjacentPairs :: [a] -> [(a,a)]
adjacentPairs xs = zip (init xs) (tail xs)
    
-- ==================================================
-- Running networks.
-- ==================================================  

-- | Feed the output from the previous neuron layer into another neuron.
feedForward :: ActivationFunction -> Vector R -> Neuron -> R
feedForward act x (Neuron bias w) = act z
    where z = (w `dot` x) + bias

-- | Run a neural network layer given input 'x'.
runLayer :: ActivationFunction -> Vector R -> Layer -> Vector R
runLayer act x (Layer neurons) = vector $ map (feedForward act x) neurons

-- | Run a neural network with the input vector 'x'.
runNetwork :: Network -> ActivationFunction -> Vector R -> Vector R
runNetwork (Network []) _act x = x
runNetwork (Network (l:ls)) act input = runNetwork (Network ls) act (runLayer act input l)


-- ==================================================
-- Activation functions.
-- ==================================================  
  
sigmoid :: ActivationFunction
sigmoid !z = 1.0 / (1.0 + exp (-z))
