{-# LANGUAGE BangPatterns #-}
{- | Parser for MNIST data.
Format: n lines of 785 double precision numbers, first 784 are the image, last is the expected number.
-}
module Main where

import           Data.Attoparsec.ByteString.Char8 (Parser, char, double, parseOnly, sepBy)
import qualified Data.ByteString.Char8 as BS
import           Numeric.LinearAlgebra (R, Vector, assoc, vector)
import qualified Data.Vector as V

data Sample = Sample
              { sampleInput          :: {-# UNPACK #-} !(Vector R)
              , sampleExpectedOutput :: {-# UNPACK #-} !(Vector R)
              } deriving (Read, Show)

makeSample :: ([R], [R]) -> Sample
makeSample (image, [expected]) = Sample (vector image) (assoc 10 0 [(floor expected, 1)])

sample :: Parser Sample
sample = do
  x <- double `sepBy` (char ' ')
  let !s = makeSample $ splitAt 784 x
  pure s

toSampleSet :: V.Vector Sample -> (V.Vector Sample, V.Vector Sample, V.Vector Sample)
toSampleSet vec = (train, validation, test)
    where (train, rest) = V.splitAt 50000 vec
          (validation, test) = V.splitAt 10000 rest

parseMnist :: BS.ByteString -> (V.Vector Sample, V.Vector Sample, V.Vector Sample)
parseMnist = toSampleSet . V.unfoldr parseLines . BS.lines
    where parseLines :: [BS.ByteString] -> Maybe (Sample, [BS.ByteString])
          parseLines []     = Nothing
          parseLines (l:ls) = Just (fromRight $ parseOnly sample l, ls)

          fromRight (Right b) = b

main :: IO ()
main = do
    !contents <- BS.readFile "/home/matt/Downloads/mnist.dat"
    let (!train, !validation, !test) = parseMnist contents
    print $
        show (length train)      ++ " " ++
        show (length validation) ++ " " ++
        show (length test)
