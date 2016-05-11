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
       ,   gradientDescent

           -- * Cost function derivatives
       ,   mse'

           -- * Internal (exposed for testing)
       ,   computeActivations
       ) where

import ML.NN.ActivationFunction (ActivationFunction, ActivationFunctionDerivative)

import Control.Monad (forM, replicateM)
import Control.Monad.Random (Rand, liftRand)
import Control.Monad.State (State, evalState, get, put)
import Data.Foldable (foldl')
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
               { networkLayers :: [Layer] -- ^ Layers in the neural network.
               } deriving (Show, Read)


-- | Configuration for training a network.
data TrainingConfig = TrainingConfig
                      { trainingEpochs :: Int  -- ^ Number of epochs of training to perform.
                      , trainingEta    :: R    -- ^ η - Learning rate.
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

-- | Generate a random neural network given the sizes of each of the layers.
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
-- The first parameter is the output activation, the second parameter is the
-- expected output.
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
instance Monoid Gradient where
    mempty = Gradient [] []
    (Gradient [] []) `mappend` g = g
    g `mappend` (Gradient [] []) = g
    (Gradient lb lw) `mappend` (Gradient rb rw) = Gradient (zipWith (+) lb rb) (zipWith (+) lw rw)

-- | Return lists of the z and activation values from each layer in the network.
--
-- The values are returned in reverse order for use by the backpropogation algorithm.
-- I chose to use the State monad so it's more explicit that the output activation of
-- one layer is the input to the next.
computeActivations :: ActivationFunction -> Network -> Vector R -> ([Vector R], [Vector R])
computeActivations act network input = (reverse zs, reverse (input:as))
    where (zs, as) = unzip $ evalState (mapM runLayer (networkLayers network)) input

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
backpropogation act act' cost' network (Sample x y) = gradient
    where (z:zs, a:a':activations) = computeActivations act network x
          (delta_L, (nablaB_L, nablaW_L)) = backpropogationLastLayer act' cost' y z a a'
          initLayers = init $ networkLayers network
          gradient =
              backpropogationHiddenLayers act' delta_L nablaB_L nablaW_L zs activations initLayers

-- | Compute delta, ∇b, and ∇w for the last (output) layer in the network.
backpropogationLastLayer :: ActivationFunctionDerivative
                         -> CostDerivative
                         -> Vector R  -- ^ Expected output.
                         -> Vector R  -- ^ zL
                         -> Vector R  -- ^ aL
                         -> Vector R  -- ^ aL-1
                         -> (Vector R, (Vector R, Matrix R))
backpropogationLastLayer act' cost' y z a a' = (delta, (nabla_b, nabla_w))
    where delta   = cost' a y * cmap act' z
          nabla_b = delta
          nabla_w = asColumn delta <> asRow a'

-- | Compute the gradient for layers 1 to L-1.
--
-- This works by scanning the layers from right to left (scanr), computing the new
-- delta (used for the next layer), and emitting ∇b and ∇w for the layer.
backpropogationHiddenLayers :: ActivationFunctionDerivative
                            -> Vector R    -- ^ delta
                            -> Vector R    -- ^ ∇b for layer L.
                            -> Matrix R    -- ^ ∇w for layer L.
                            -> [Vector R]  -- ^ zs.
                            -> [Vector R]  -- ^ activations.
                            -> [Layer]     -- ^ Layers 1 to L-1.
                            -> Gradient
backpropogationHiddenLayers act' delta_L nablaB_L nablaW_L zs activations layers =
    uncurry Gradient $ unzip $ fmap snd output
    where
        output :: [(Vector R, (Vector R, Matrix R))]  -- ^ (delta, (∇b, ∇w))
        output = scanr computeNabla (delta_L, (nablaB_L, nablaW_L)) (zip3 layers zs activations)

        computeNabla :: (Layer, Vector R, Vector R)       -- ^ (layer, z, activation)
                     -> (Vector R, (Vector R, Matrix R))  -- ^ (delta, (∇b, ∇w))
                     -> (Vector R, (Vector R, Matrix R))  -- ^ (delta', (∇b', ∇w'))
        computeNabla (layer, z, a) (d, (_, _)) = (d', (nb', nw'))
            where sp  = cmap act' z
                  w   = layerWeights layer
                  d'  = (tr w #> d) * sp
                  nb' = d'
                  nw' = asColumn d' <> asRow a

-- | Train the neural network using gradient descent.
gradientDescent :: TrainingConfig
                -> [Sample]  -- ^ Training data (input, expected output).
                -> Network   -- ^ Current network.
                -> Network   -- ^ Updated network.
gradientDescent (TrainingConfig epochs eta act act' cost') trainingData inputNetwork =
    go epochs inputNetwork
    where
        go :: Int -> Network -> Network
        go 0     network = network
        go epoch network = go (epoch-1) (Network layers')
            where sampleGradients :: [Gradient]
                  sampleGradients = fmap (backpropogation act act' cost' network) trainingData

                  Gradient nablaB nablaW = mconcat sampleGradients

                  numSamples = fromIntegral $ length trainingData
                  layers' = zipWith3 updateWeightsAndBiases (networkLayers network) nablaB nablaW

                  updateWeightsAndBiases :: Layer -> Vector R -> Matrix R -> Layer
                  updateWeightsAndBiases (Layer b w) nb nw =
                      Layer (b-(konst (eta/numSamples) (size b)) * nb)
                            (w-(konst (eta/numSamples) (size w)) * nw)


-- ==================================================
-- Cost function derivatives.
-- ==================================================

-- | The derivative of the mean squared error cost function.
mse' :: CostDerivative
mse' outputActivations y = outputActivations - y
