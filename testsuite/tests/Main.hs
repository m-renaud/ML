module Main where

import Test.Tasty (defaultMain, testGroup)

import qualified NN.Tests

main :: IO ()
main = defaultMain $ testGroup "Tests"
    [
        NN.Tests.tests
    ]
