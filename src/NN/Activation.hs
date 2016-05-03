{-# LANGUAGE BangPatterns #-}
{- |
Activation functions for use with neural networks.
-}
module NN.Activation (sigmoid) where

import NN.ActivationFunction (ActivationFunction)

-- | The sigmoid function: https://en.wikipedia.org/wiki/Sigmoid_function.
sigmoid :: ActivationFunction
sigmoid !z = 1.0 / (1.0 + exp (-z))
