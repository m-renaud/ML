module Control.Monad.Random.Extended (shuffle, vshuffle) where

import           Control.Monad (forM_)
import           Control.Monad.Random (Rand, getRandomRs)
import           Data.Array.ST (runSTArray)
import qualified Data.Vector as V
import           GHC.Arr (elems, listArray, readSTArray, thawSTArray, writeSTArray)
import           System.Random (RandomGen)

shuffle :: RandomGen g => [a] -> Rand g [a]
shuffle xs = do
    let l = length xs
    rands <- take l <$> getRandomRs (0, l-1)
    let ar' = runSTArray $ do
            ar <- thawSTArray $ listArray (0, l-1) xs
            forM_ (zip [0..(l-1)] rands) $ \(i, j) -> do
                vi <- readSTArray ar i
                vj <- readSTArray ar j
                writeSTArray ar j vi
                writeSTArray ar i vj
            pure ar
    pure $ elems ar'

vshuffle :: RandomGen g => V.Vector a -> Rand g (V.Vector a)
vshuffle arr = V.fromList <$> shuffle (V.toList arr)
