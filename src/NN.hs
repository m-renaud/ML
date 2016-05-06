{-# LANGUAGE BangPatterns #-}
{- |
Simple neural network implementation, for learning purposes only.
-}
module NN
       (
           -- * Data types
           ActivationFunction  -- Re-export from NN.ActivationFunction.
       ,   Layer(..)
       ,   Network(..)

           -- * Network initialization
       ,   randLayer
       ,   randNetwork

           -- * Running networks
       ,   feedForward
       ,   runNetwork
       ) where

import NN.ActivationFunction (ActivationFunction)

import Control.Monad (forM, replicateM)
import Control.Monad.Random (Rand, liftRand)
import Data.Foldable (foldl')

import Data.Random.Normal (normal)
import Numeric.LinearAlgebra (Matrix, R, Vector, (><), (#>), cmap, vector)
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
             } deriving (Show, Read)

-- | A network is a list of layers.
data Network = Network
               { networkLayers :: [Layer] -- ^ Layers in the neural network.
               } deriving (Show, Read)


-- ==================================================
-- Network initialization.
-- ==================================================

-- | Generate a randomly initialized layer in a neural network.
randLayer :: RandomGen g
             => Int           -- ^ Number of inputs into the layer.
             -> Int           -- ^ Number of neurons in the layer.
             -> Rand g Layer  -- ^ Randomly generated layer.
randLayer numInputs numNeurons = do
    bias <- vector <$> replicateM numNeurons (liftRand normal)
    weights <- (numNeurons><numInputs) <$> replicateM (numNeurons*numInputs) (liftRand normal)
    return $ Layer bias weights

-- | Generate a random neural network given the sizes of each of the layers.
--
-- For example, [3,2,4] will generate a neural network with 3 input
-- neurons, 1 hidden layer with 2 neurons and 4 output neurons.
randNetwork :: RandomGen g
               => [Int]           -- ^ Number of neurons in each layer.
               -> Rand g Network  -- ^ Randomly generated network.
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
            -> Vector R            -- ^ Input to the layer.
            -> Layer               -- ^ Layer to run.
            -> Vector R            -- ^ Output from the layer.
feedForward act x (Layer biases weights) = cmap act z
    where z = (weights #> x) + biases

-- | Run a neural network.
runNetwork :: Network             -- ^ Network to run.
           -> ActivationFunction  -- ^ Neuron activation function.
           -> Vector R            -- ^ Network input.
           -> Vector R            -- ^ Network output.
runNetwork (Network layers) act input = foldl' (feedForward act) input layers
