module Main where

import ML.NN
import ML.NN.Activation (sigmoid)

import Control.Monad.Random (evalRandIO)
import Numeric.LinearAlgebra (vector)

main :: IO ()
main = do
  net <- evalRandIO $ randNetwork [256, 40, 10]
  print $ runNetwork net sigmoid (vector $ concat $ replicate 32 [1,5,4,3,6,5,4,2])
