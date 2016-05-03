module NN.ActivationFunction (ActivationFunction) where

import Numeric.LinearAlgebra (R)

-- | The activation function maps R -> R.
type ActivationFunction = R -> R
