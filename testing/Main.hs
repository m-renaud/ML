{-# LANGUAGE BangPatterns #-}
module Main where

import           ML.MnistLoader (parseMnist)
import           ML.NN
import           ML.NN.Activation (sigmoid, sigmoid')

import           Control.Monad.Random (evalRandIO)
import qualified Data.ByteString.Char8 as BS
import qualified Data.Vector as V


config :: TrainingConfig
config = TrainingConfig 3.0 sigmoid sigmoid' mse'

main :: IO ()
main = do
  !content <- BS.readFile "/home/matt/Downloads/mnist.dat"
  let (!training, !_validation, !_test) = parseMnist content
      !input1 = sampleInput $ V.head training
  !net <- evalRandIO $ randNetwork [784, 30, 10]
  print $ runNetwork net sigmoid input1
  -- let !net' = gradientDescentCore config (V.take 8000 training) net
  !net' <- sgd config 1 10 (V.take 8000 training) net
  print $ runNetwork net' sigmoid input1
  -- !net'' <- sgd config 2 10 (V.take 1000 $ V.drop 1000 training) net'
  -- print $ runNetwork net'' sigmoid input1
  -- !net'' <- sgd config 2 100 training net'
  -- print $ runNetwork net'' sigmoid input1
  --
