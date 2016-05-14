module Main where

import           Data.Attoparsec.ByteString.Char8 hiding (take)
import qualified Data.ByteString as BS
import           Numeric.LinearAlgebra (Vector, R, vector)

type Sample = (Vector R, Int)

makeSample :: ([R], [R]) -> Sample
makeSample (image, [expected]) = (vector image, floor expected)

sampleSet :: Parser ([Sample], [Sample], [Sample])
sampleSet = (,,) <$> count 50000 sample <*> count 10000 sample <*> count 10000 sample

sample :: Parser Sample
sample = makeSample . splitAt 784 <$> double `sepBy` (char ' ') <* endOfLine

main :: IO ()
main = do
    contents <- BS.readFile "/home/matt/Downloads/mnist.dat"
    case parseOnly sampleSet contents of
        Left err -> print err
        Right (train, validation, test) -> print $
            show (length train)      ++ " " ++
            show (length validation) ++ " " ++
            show (length test)
