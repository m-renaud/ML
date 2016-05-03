{- |
Extended linear algebra library containing utilities not present in the hmatrix package.
-}
module Numeric.LinearAlgebra.Extended (vectorize) where

import Foreign.Storable (Storable)
import Numeric.LinearAlgebra (Vector, fromList, toList)

-- | Vectorize a function.
--
-- /Note: This is VERY inefficient as currently implemented./
vectorize :: Storable t => (t -> t) -> Vector t -> Vector t
vectorize f = fromList . map f . toList
