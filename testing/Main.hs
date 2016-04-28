module Main where

import NN

import Control.Monad.Random (evalRandIO)
import Numeric.LinearAlgebra (vector)

main :: IO ()
main = do
  net <- evalRandIO $ randNetwork [100, 4000, 200]
  print $ runNetwork net sigmoid (vector $ concat $ replicate 10 [1,5,4,3,6,5,4,2,3,4])
