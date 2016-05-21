{- |
Module:      : ML.CostFunctions
Description  : Cost function derivatives for use in machine learning algorithms.
-}
module ML.CostFunctions
       (
           -- * Data types
           CostDerivative

           -- * Cost function derivatives
       ,   mse') where

import Numeric.LinearAlgebra (R, Vector)

-- | A vectorized function which returns ∂Cₓ/∂a.
--
-- The first parameter is the output activation, the second parameter
-- is the expected output.
type CostDerivative = Vector R -> Vector R -> Vector R

-- | The derivative of the mean squared error cost function.
mse' :: CostDerivative
mse' outputActivations y = outputActivations - y
