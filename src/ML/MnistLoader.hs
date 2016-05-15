{-# LANGUAGE BangPatterns #-}
module ML.MnistLoader (parseMnist) where

import ML.Sample (Sample(..))

import           Data.Attoparsec.ByteString.Char8 (Parser, char, double, parseOnly, sepBy)
import qualified Data.ByteString.Char8 as BS
import           Numeric.LinearAlgebra (R, assoc, vector)
import qualified Data.Vector as V

makeSample :: ([R], [R]) -> Sample
makeSample (image, [expected]) = Sample (vector image) (assoc 10 0 [(floor expected, 1)])

toSampleSet :: V.Vector Sample -> (V.Vector Sample, V.Vector Sample, V.Vector Sample)
toSampleSet vec = (train, validation, test)
    where (train, rest) = V.splitAt 50000 vec
          (validation, test) = V.splitAt 10000 rest

sample :: Parser Sample
sample = do
  x <- double `sepBy` (char ' ')
  let !s = makeSample $ splitAt 784 x
  pure s

parseMnist :: BS.ByteString -> (V.Vector Sample, V.Vector Sample, V.Vector Sample)
parseMnist = toSampleSet . V.unfoldr parseLines . BS.lines
    where parseLines :: [BS.ByteString] -> Maybe (Sample, [BS.ByteString])
          parseLines []     = Nothing
          parseLines (l:ls) = Just (fromRight $ parseOnly sample l, ls)

          fromRight (Right b) = b
