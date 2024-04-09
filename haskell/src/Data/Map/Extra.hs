{-# LANGUAGE FlexibleInstances #-}

module Data.Map.Extra where

import qualified Data.Map as Map
import Data.Foldable (toList)

--------------------------------------------------
-- * MatchAny

-- | A "wildcard" match, useful for doing lookups in 'Map's where the key might
-- contain `Any a`.
data Any a = A !a | Any
  deriving (Eq, Ord, Show)

-- | Search a 'Map' where elements in the @k@'s structure can have wildcard
-- values
lookupMatchAny :: MatchAny k => k -> Map.Map k v -> Maybe v
lookupMatchAny key = Map.foldrWithKey go Nothing
  where
    go k v acc
      | matchAny key k = Just v
      | otherwise = acc

class MatchAny a where
    matchAny :: a -> a -> Bool

instance Eq a => MatchAny (Any a) where
    matchAny (A x) (A y) = x == y
    matchAny Any _ = True
    matchAny _ Any = True

instance (MatchAny a, MatchAny b) => MatchAny (a, b) where
    matchAny (a1, b1) (a2, b2) = matchAny a1 a2 && matchAny b1 b2

instance (MatchAny a, MatchAny b, MatchAny c) => MatchAny (a, b, c) where
    matchAny (a1, b1, c1) (a2, b2, c2) = matchAny a1 a2 && matchAny b1 b2 && matchAny c1 c2

instance (MatchAny a) => MatchAny (Maybe a) where
    matchAny (Just a) (Just b) = matchAny a b
    matchAny Nothing Nothing = True
    matchAny _ _ = False

-- instance {-# OVERLAPPABLE #-} (Foldable t, MatchAny a) => MatchAny (t a) where
--     matchAny t1 t2 = and $ zipWith matchAny (toList t1) (toList t2)
