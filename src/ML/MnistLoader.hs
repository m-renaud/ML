{-# LANGUAGE BangPatterns #-}
{-|
Provides a parser for text formatted MNIST data.
-}
module ML.MnistLoader (parseMnist) where

import           ML.Sample (Sample(..))

import           Data.Attoparsec.ByteString.Char8 (Parser, char, double, parseOnly, sepBy)
import qualified Data.ByteString.Char8 as BS
import qualified Data.Vector as V
import           Numeric.LinearAlgebra (R, assoc, vector)

makeSample :: ([R], [R]) -> Sample
makeSample (image, [expected]) = Sample (vector image) (assoc 10 0 [(floor expected, 1)])

-- | Break the list of samples into components.
--
-- These components are implicit in the file format.
toSampleSet :: V.Vector Sample -> (V.Vector Sample, V.Vector Sample, V.Vector Sample)
toSampleSet vec = (train, validation, test)
    where (train, rest) = V.splitAt 50000 vec
          (validation, test) = V.splitAt 10000 rest

-- | Parse an MNIST sample image.
sample :: Parser Sample
sample = do
    !x <- double `sepBy` (char ' ')
    let !s = makeSample $ splitAt 784 x
    pure s

-- | Parse the given bytestring into (training data, validation data, test data).
--
-- Data format:
--
--   * 70000 lines lines of 785 doubles.
--   * first 1     - 50000 lines are test data.
--   * lines 50001 - 60000 are validation data.
--   * lines 60001 - 70000 are test data.
--   * first 784 double of each line are the image.
--   * last double of each is the expected value.
parseMnist :: BS.ByteString -> (V.Vector Sample, V.Vector Sample, V.Vector Sample)
parseMnist = toSampleSet . V.unfoldr parseLines . BS.lines
    where parseLines :: [BS.ByteString] -> Maybe (Sample, [BS.ByteString])
          parseLines []     = Nothing
          parseLines (l:ls) = Just (fromRight $ parseOnly sample l, ls)

          fromRight (Right b) = b
