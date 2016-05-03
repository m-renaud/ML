{-# LANGUAGE BangPatterns #-}
{- |
Neural network implementation.
-}
module NN
       (
         -- * Data types
         Layer(..)
       , Network(..)

         -- * Network initialization
       , randLayer
       , randNetwork

         -- * Running networks
       , feedForward
       , runNetwork
       ) where

import NN.ActivationFunction (ActivationFunction)

import Control.Monad (forM, replicateM)
import Control.Monad.Random (Rand, liftRand)
import Data.Random.Normal (normal)
import Numeric.LinearAlgebra (Matrix, R, Vector, (><), (#>), vector)
import Numeric.LinearAlgebra.Extended (vectorize)
import System.Random (RandomGen)


-- ==================================================
-- Types.
-- ==================================================

-- | A layer in a neural network is a bias vector and weight matrix.
--
-- Let /n/ be the number of neurons in the layer and /m/ be the number
-- of inputs to the layer. Then 'layerBiases' is a vector in Rⁿ and
-- 'layerWeights' is an /nxm/ real matrix.
data Layer = Layer
             { layerBiases  :: Vector R  -- ^ A vector in Rⁿ representing the neuron biases.
             , layerWeights :: Matrix R  -- ^ An /nxm/ real matrix of the neuron weights.
             } deriving Show

-- | A network is a list of layers.
data Network = Network
               { networkLayers :: [Layer] -- ^ The layers in the neural network.
               } deriving Show


-- ==================================================
-- Network initialization.
-- ==================================================

-- | Generate a randomly initialized layer in a neural network.
randLayer :: RandomGen g
             => Int  -- ^ The number of inputs into the layer.
             -> Int  -- ^ The number of neurons in the layer.
             -> Rand g Layer
randLayer numInputs numNeurons = do
    bias <- vector <$> replicateM numNeurons (liftRand normal)
    weights <- (numNeurons><numInputs) <$> replicateM (numNeurons*numInputs) (liftRand normal)
    return $ Layer bias weights

-- | Generate a random neural network given the sizes of each of the layers.
--
-- For example, [3,2,4] will generate a neural network with 3 input
-- neurons, 1 hidden layer with 2 neurons and 4 output neurons.
randNetwork :: RandomGen g
               => [Int]  -- ^ The number of neurons in each layer.
               -> Rand g Network
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

-- | Feed the output from a previous layer to the next layer.
feedForward :: ActivationFunction  -- ^ Neuron activation function.
               -> Vector R         -- ^ Input to the layer.
               -> Layer            -- ^ The layer to run.
               -> Vector R
feedForward act x (Layer biases weights) = (vectorize act) z
    where z = (weights #> x) + biases

-- | Run a neural network with the input vector 'x'.
runNetwork :: Network -> ActivationFunction -> Vector R -> Vector R
runNetwork (Network []) _act x = x
runNetwork (Network (l:ls)) act input = runNetwork (Network ls) act (feedForward act input l)
