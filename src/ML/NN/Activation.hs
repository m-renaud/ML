{-# LANGUAGE BangPatterns #-}
{- |
Activation functions and their derivatives for use with neural networks.
-}
module ML.NN.Activation (sigmoid, sigmoid') where

import ML.NN.ActivationFunction (ActivationFunction, ActivationFunctionDerivative)

-- | The sigmoid function: https://en.wikipedia.org/wiki/Sigmoid_function.
sigmoid :: ActivationFunction
sigmoid !z = 1.0 / (1.0 + exp (-z))

-- | The derivative of the sigmoid function.
sigmoid' :: ActivationFunctionDerivative
sigmoid' !z = sz * (1 - sz)
    where !sz = sigmoid z
