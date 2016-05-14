{-# LANGUAGE BangPatterns #-}
{- |
Simple neural network implementation, for learning purposes only.
-}
module ML.NN
       (
           -- * Data types
           ActivationFunction  -- Re-export from NN.ActivationFunction.
       ,   ActivationFunctionDerivative  -- Re-export from NN.ActivationFunction.
       ,   Layer(..)
       ,   Network(..)

           -- * Network initialization
       ,   randLayer
       ,   randNetwork

           -- * Running networks
       ,   feedForward
       ,   runNetwork

           -- * Training networks
       ,   TrainingConfig(..)
       ,   Sample(..)
       ,   Gradient(..)
       ,   CostDerivative
       ,   sgd

           -- * Cost function derivatives
       ,   mse'

           -- * Internal (exposed for testing)
       ,   computeActivations
       ) where

import ML.NN.ActivationFunction (ActivationFunction, ActivationFunctionDerivative)

import Control.Monad (forM, replicateM)
import Control.Monad.Random (Rand, liftRand)
import Control.Monad.Random.Extended (shuffle)
import Control.Monad.State (State, evalState, get, put)
import Data.Foldable (foldl')
import Data.List.Split (chunksOf)
import Data.Random.Normal (normal)
import Numeric.LinearAlgebra (Matrix, R, Vector, (><), (<>), (#>), cmap, konst, tr, vector)
import Numeric.LinearAlgebra.Data (asColumn, asRow, size)
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
               { networkLayers :: [Layer]
               } deriving (Show, Read)


-- | Configuration for training a network.
data TrainingConfig = TrainingConfig
                      { trainingEta :: R  -- ^ η - Learning rate.
                      , trainingActivation :: ActivationFunction
                      , trainingActivationDerivative :: ActivationFunctionDerivative
                      , trainingCostDerivative :: CostDerivative
                      }

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

-- | Generate a random neural network given the size of each layer.
--
-- For example, @[3,2,4]@ will generate a neural network with 3 input
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


-- ==================================================
-- Training networks.
-- ==================================================

-- | A vectorized function which returns ∂Cₓ/∂a.
--
-- The first parameter is the output activation, the second parameter
-- is the expected output.
type CostDerivative = Vector R -> Vector R -> Vector R

-- | A training sample.
data Sample = Sample
              { sampleInput          :: Vector R
              , sampleExpectedOutput :: Vector R
              } deriving (Read, Show)

-- | The gradient of the network.
data Gradient = Gradient
                { gradientNablaB :: [Vector R]
                , gradientNablaW :: [Matrix R]
                } deriving (Read, Show)

-- | The monoid instance for gradient sums the components.
--
-- We treat a gradiant with no bias or weight values as the `mempty`
-- Gradient.
instance Monoid Gradient where
    mempty = Gradient [] []
    (Gradient [] []) `mappend` g = g
    g `mappend` (Gradient [] []) = g
    (Gradient lb lw) `mappend` (Gradient rb rw) =
        Gradient (zipWith (+) lb rb) (zipWith (+) lw rw)

-- | Return z and activation values for each layer in the network.
--
-- The values are returned in reverse order for use by the
-- backpropogation algorithm.  We the State monad so it's more
-- explicit that the output activation of one layer is the input to
-- the next.
computeActivations :: ActivationFunction -> Network -> Vector R -> ([Vector R], [Vector R])
computeActivations act (Network layers) input = (reverse zs, reverse (input:as))
    where (zs, as) = unzip $ evalState (mapM runLayer layers) input

          runLayer :: Layer -> State (Vector R) (Vector R, Vector R)
          runLayer (Layer b w) = do
              x <- get
              let z = w #> x + b
                  a = cmap act z
              put a
              pure $ (z,a)

-- | Compute the gradient for the training example.
backpropogation :: ActivationFunction
                -> ActivationFunctionDerivative
                -> CostDerivative
                -> Network
                -> Sample
                -> Gradient
backpropogation act act' cost' network@(Network layers) (Sample x y) =
    Gradient (reverse $ nablaB_L:nablaBs) (reverse $ nablaW_L:nablaWs)
    where (z:zs, a:a':as) = computeActivations act network x
          delta_L = cost' a y * cmap act' z

          nablaB_L = delta_L
          nablaW_L = asColumn delta_L <> asRow a'
          (nablaBs, nablaWs) =
              backpropogationBackwardsPass act' delta_L zs as (reverse $ tail layers)

-- | Perform the backwards pass of the backpropogation algorithm.
--
-- Traverse the previously reverse z values, activations, and layers
-- from layer L to 2.  We thread the delta value along as state
backpropogationBackwardsPass :: ActivationFunctionDerivative
                             -> Vector R    -- ^ delta
                             -> [Vector R]  -- ^ zs in reverse order.
                             -> [Vector R]  -- ^ activations in reverse order.
                             -> [Layer]     -- ^ Layers L to 2.
                             -> ([Vector R], [Matrix R])  -- ^
backpropogationBackwardsPass act' delta_L zs as layers = unzip output
    where output = evalState (mapM processLayer (zip3 layers zs as)) delta_L

          processLayer :: (Layer, Vector R, Vector R) -> State (Vector R) (Vector R, Matrix R)
          processLayer ((Layer _bias weights), z, a) = do
              delta <- get
              let actPrime = cmap act' z
                  delta'   = (tr weights #> delta) * actPrime
              put delta'
              pure (delta', asColumn delta' <> asRow a)

-- | Update the network's weights and biases by applying gradient
-- descent for the given sample input.
gradientDescentCore :: TrainingConfig -> [Sample] -> Network -> Network
gradientDescentCore (TrainingConfig eta act act' cost') trainingData net@(Network layers) =
    Network layers'
    where sampleGradients :: [Gradient]
          sampleGradients = fmap (backpropogation act act' cost' net) trainingData

          Gradient nablaB nablaW = mconcat sampleGradients

          numSamples = fromIntegral $ length trainingData
          layers' = zipWith3 updateWeightsAndBiases layers nablaB nablaW

          updateWeightsAndBiases :: Layer -> Vector R -> Matrix R -> Layer
          updateWeightsAndBiases (Layer b w) nb nw =
              Layer (b-(konst (eta/numSamples) (size b)) * nb)
                    (w-(konst (eta/numSamples) (size w)) * nw)

-- | Train the neural network using stochastic gradient descent.
sgd :: RandomGen g
       => TrainingConfig
       -> Int  -- ^ epochs
       -> Int  -- ^ mini-batch size
       -> [Sample]
       -> Network
       -> Rand g Network
sgd trainingConfig epochs miniBatchSize trainingData network = go epochs network
    where go 0     net = pure net
          go epoch net = do
              shuffledTrainingData <- shuffle trainingData
              let miniBatches = miniBatchSize `chunksOf` shuffledTrainingData
                  net' = foldr (gradientDescentCore trainingConfig) net miniBatches
              go (epoch-1) net'


-- ==================================================
-- Cost function derivatives.
-- ==================================================

-- | The derivative of the mean squared error cost function.
mse' :: CostDerivative
mse' outputActivations y = outputActivations - y
