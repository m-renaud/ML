{-# LANGUAGE BangPatterns #-}
module ML.Sample (Sample(..)) where

import Numeric.LinearAlgebra (R, Vector)

-- | A training sample.
data Sample = Sample
              { sampleInput          :: {-# UNPACK #-} !(Vector R)
              , sampleExpectedOutput :: {-# UNPACK #-} !(Vector R)
              } deriving (Read, Show)
