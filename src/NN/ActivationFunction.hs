module NN.ActivationFunction (ActivationFunction) where

import Numeric.LinearAlgebra (R)

-- | An activation function maps R -> R.
type ActivationFunction = R -> R
