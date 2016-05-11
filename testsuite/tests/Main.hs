module Main where

import Test.Tasty (defaultMain, testGroup)

import qualified ML.NN.Tests

main :: IO ()
main = defaultMain $ testGroup "Tests"
    [
        ML.NN.Tests.tests
    ]
