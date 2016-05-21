{-# LANGUAGE BangPatterns #-}
module ML.NN.Training
       (
           -- * Training networks
           TrainingConfig(..)
       ,   Sample(..)
       ,   Gradient(..)
       ,   sgd
       ,   gradientDescentCore

           -- * Internal (exposed for testing)
       ,   computeZsAndAs
       ) where

import           ML.CostFunctions (CostDerivative, mse')
import           ML.NN
import           ML.NN.ActivationFunction (ActivationFunctionDerivative)
import           ML.Sample (Sample(..))

import           Control.Monad.Random (evalRandIO)
import           Control.Monad.Random.Extended (vshuffle)
import           Control.Monad.State (State, evalState, get, put)
import           Data.Foldable (foldl')
import           Data.List (unfoldr)
import qualified Data.Vector as V
import           Numeric.LinearAlgebra (Matrix, R, Vector, (#>), (<>), asColumn, asRow, cmap, konst, outer, size, tr)

-- | Configuration for training a network.
data TrainingConfig = TrainingConfig
                      { trainingEta :: R  -- ^ η - Learning rate.
                      , trainingActivation :: ActivationFunction
                      , trainingActivationDerivative :: ActivationFunctionDerivative
                      , trainingCostDerivative :: CostDerivative
                      }

-- | The gradient of the network.
data Gradient = Gradient
                { gradientNablaB :: {-# UNPACK #-} !(V.Vector (Vector R))
                , gradientNablaW :: {-# UNPACK #-} !(V.Vector (Matrix R))
                } deriving (Read, Show)

sumGradients :: V.Vector Gradient -> Gradient
sumGradients = V.foldl1' addGradient
    where (Gradient !lb !lw) `addGradient` (Gradient !rb !rw) =
              Gradient (V.zipWith (+) lb rb) (V.zipWith (+) lw rw)

-- | Return z and activation values for each layer in the network.
--
-- The values are returned in reverse order for use by the
-- backpropogation algorithm.  We the State monad so it's more
-- explicit that the output activation of one layer is the input to
-- the next.
computeZsAndAs :: ActivationFunction -> Network -> Vector R -> ([Vector R], [Vector R])
computeZsAndAs act (Network layers) input = (reverse zs, reverse (input:as))
    where (zs, as) = unzip $ evalState (mapM computeZA layers) input

          computeZA :: Layer -> State (Vector R) (Vector R, Vector R)
          computeZA (Layer b w) = do
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
backpropogation act act' cost' !net@(Network !layers) (Sample !x !y) =
    Gradient (V.reverse $ V.cons nablaB_L nablaBs) (V.reverse $ V.cons nablaW_L nablaWs)
    where (z:zs, a:a':as) = computeZsAndAs act net x
          delta_L  = cost' a y * cmap act' z
          nablaB_L = delta_L
          nablaW_L = asColumn delta_L <> asRow a'
          (nablaBs, nablaWs) =
              backpropogationBackwardsPass act' delta_L zs as (reverse $ tail layers)

-- | Perform the backwards pass of the backpropogation algorithm.
--
-- Traverse the previously reverse z values, activations, and layers
-- from layer L to 2.  We thread the delta value along as state
backpropogationBackwardsPass :: ActivationFunctionDerivative
                             -> Vector R    -- ^ delta for the last layer.
                             -> [Vector R]  -- ^ zs in reverse order.
                             -> [Vector R]  -- ^ activations in reverse order.
                             -> [Layer]     -- ^ Layers L to 2.
                             -> (V.Vector (Vector R), V.Vector (Matrix R))  -- ^
backpropogationBackwardsPass act' delta_L zs as layers = V.unzip output
    where !output = V.fromList $ evalState (mapM processLayer (zip3 layers zs as)) delta_L

          processLayer :: (Layer, Vector R, Vector R) -> State (Vector R) (Vector R, Matrix R)
          processLayer ((Layer _bias weights), z, a) = do
              delta <- get
              let actPrime = cmap act' z
                  delta'   = (tr weights #> delta) * actPrime
              put delta'
              pure (delta', delta' `outer` a)

-- | Update the network's weights and biases by applying gradient
--   descent for the given sample input.
gradientDescentCore :: TrainingConfig -> V.Vector Sample -> Network -> Network
gradientDescentCore (TrainingConfig eta act act' cost') !trainingData !net@(Network !layers) =
    Network layers'
    where sampleGradients :: V.Vector Gradient
          !sampleGradients = V.map (backpropogation act act' cost' net) trainingData

          Gradient !nablaB !nablaW = sumGradients sampleGradients

          !layers' = V.toList $ V.zipWith3 updateWeightsAndBiases (V.fromList layers) nablaB nablaW

          updateWeightsAndBiases :: Layer -> Vector R -> Matrix R -> Layer
          updateWeightsAndBiases (Layer b w) nb nw =
              Layer (b-(konst (eta/numSamples) (size b)) * nb)
                    (w-(konst (eta/numSamples) (size w)) * nw)
              where numSamples = fromIntegral $ length trainingData


-- | Train the neural network using stochastic gradient descent.
sgd :: TrainingConfig
    -> Int  -- ^ epochs
    -> Int  -- ^ mini-batch size
    -> V.Vector Sample
    -> Network
    -> IO Network
sgd trainingConfig !epochs !miniBatchSize !trainingData !network = trainEpoch 0 network
    where trainEpoch epoch !net
              | epoch == epochs = putStrLn "done!" >> pure net
              | otherwise = do
                    !shuffledTrainingData <- evalRandIO $ vshuffle trainingData
                    let miniBatches = miniBatchSize `vChunksOf` shuffledTrainingData
                        !net' = foldl' (flip (gradientDescentCore trainingConfig)) net miniBatches
                    putStrLn $ "Completed epoch: " ++ show epoch
                    trainEpoch (epoch+1) net'

vChunksOf :: Int -> V.Vector a -> [V.Vector a]
vChunksOf n vec = Data.List.unfoldr makeChunk vec
    where makeChunk :: V.Vector a -> Maybe (V.Vector a, V.Vector a)
          makeChunk v | V.null v  = Nothing
                      | otherwise = Just $ V.splitAt n v
