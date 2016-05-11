{-|
Type aliases for activation functions and their derivative.
-}
module ML.NN.ActivationFunction (ActivationFunction, ActivationFunctionDerivative) where

import Numeric.LinearAlgebra (R)

-- | An activation function for a neuron.
type ActivationFunction = R -> R

-- | The derivative of a neuron activation function.
type ActivationFunctionDerivative = R -> R
