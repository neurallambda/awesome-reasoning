{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverlappingInstances #-}

module Data.Map.Extra where

import qualified Data.Map as Map

--------------------------------------------------
-- * MatchAny

-- | A "wildcard" match, useful for doing lookups in 'Map's where the key might
-- contain `Any a`.
data Any a = A !a | Any
  deriving (Eq, Ord, Show)

-- | Search a 'Map' where elements in the @k@'s structure can have wildcard
-- values
lookupMatchAny :: MatchAny k => k -> Map.Map k v -> Maybe v
lookupMatchAny key myMap = Map.foldrWithKey go Nothing myMap
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

instance Eq a => MatchAny (Maybe (Any a)) where
    matchAny (Just x) (Just y) = matchAny x y
    matchAny x y = x == y


----------
-- ** 2-tuples

instance (MatchAny a, MatchAny b) => MatchAny (a, b) where
    matchAny (a1, b1) (a2, b2) = matchAny a1 a2 && matchAny b1 b2

instance (MatchAny a, MatchAny b) => MatchAny (Any a, b) where
    matchAny (Any, b1) (_, b2) = matchAny b1 b2
    matchAny (A a1, b1) (A a2, b2) = matchAny a1 a2 && matchAny b1 b2
    matchAny _ _ = False

instance (MatchAny a, MatchAny b) => MatchAny (a, Any b) where
    matchAny (a1, Any) (a2, _) = matchAny a1 a2
    matchAny (a1, A b1) (a2, A b2) = matchAny a1 a2 && matchAny b1 b2
    matchAny _ _ = False

instance (MatchAny a, MatchAny b) => MatchAny (Any a, Any b) where
    matchAny (Any, Any) _ = True
    matchAny (A a1, Any) (A a2, _) = matchAny a1 a2
    matchAny (Any, A b1) (_, A b2) = matchAny b1 b2
    matchAny (A a1, A b1) (A a2, A b2) = matchAny a1 a2 && matchAny b1 b2
    matchAny _ _ = False


----------
-- ** 3-tuples

instance (MatchAny a, MatchAny b, MatchAny c) => MatchAny (a, b, c) where
    matchAny (a1, b1, c1) (a2, b2, c2) = matchAny a1 a2 && matchAny b1 b2 && matchAny c1 c2

instance (MatchAny a, MatchAny b, MatchAny c) => MatchAny (Any a, b, c) where
    matchAny (Any, b1, c1) (_, b2, c2) = matchAny b1 b2 && matchAny c1 c2
    matchAny (_, b1, c1) (_, b2, c2) = matchAny b1 b2 && matchAny c1 c2
    matchAny (A a1, b1, c1) (A a2, b2, c2) = matchAny a1 a2 && matchAny b1 b2 && matchAny c1 c2
    matchAny _ _ = False

instance (MatchAny a, MatchAny b, MatchAny c) => MatchAny (a, Any b, c) where
    matchAny (a1, Any, c1) (a2, _, c2) = matchAny a1 a2 && matchAny c1 c2
    matchAny (a1, _, c1) (a2, Any, c2) = matchAny a1 a2 && matchAny c1 c2
    matchAny (a1, A b1, c1) (a2, A b2, c2) = matchAny a1 a2 && matchAny b1 b2 && matchAny c1 c2
    matchAny _ _ = False

instance (MatchAny a, MatchAny b, MatchAny c) => MatchAny (a, b, Any c) where
    matchAny (a1, b1, Any) (a2, b2, _) = matchAny a1 a2 && matchAny b1 b2
    matchAny (a1, b1, _) (a2, b2, Any) = matchAny a1 a2 && matchAny b1 b2
    matchAny (a1, b1, A c1) (a2, b2, A c2) = matchAny a1 a2 && matchAny b1 b2 && matchAny c1 c2
    matchAny _ _ = False

instance (MatchAny a, MatchAny b, MatchAny c) => MatchAny (Any a, Any b, c) where
    matchAny (Any, Any, c1) (_, _, c2) = matchAny c1 c2
    matchAny (A a1, Any, c1) (A a2, _, c2) = matchAny a1 a2 && matchAny c1 c2
    matchAny (Any, A b1, c1) (_, A b2, c2) = matchAny b1 b2 && matchAny c1 c2

    matchAny (_, _, c1) (Any, Any, c2) = matchAny c1 c2
    matchAny (A a1, _, c1) (A a2, Any, c2) = matchAny a1 a2 && matchAny c1 c2
    matchAny (_, A b1, c1) (Any, A b2, c2) = matchAny b1 b2 && matchAny c1 c2

    matchAny (Any, _, c1) (_, Any, c2) = matchAny c1 c2
    matchAny (_, Any, c1) (Any, _, c2) = matchAny c1 c2

    matchAny (A a1, A b1, c1) (A a2, A b2, c2) = matchAny a1 a2 && matchAny b1 b2 && matchAny c1 c2
    matchAny _ _ = False

instance (MatchAny a, MatchAny b, MatchAny c) => MatchAny (Any a, b, Any c) where
    matchAny (Any, b1, Any) (_, b2, _) = matchAny b1 b2
    matchAny (A a1, b1, Any) (A a2, b2, _) = matchAny a1 a2 && matchAny b1 b2
    matchAny (Any, b1, A c1) (_, b2, A c2) = matchAny b1 b2 && matchAny c1 c2

    matchAny (_, b1, _) (Any, b2, Any) = matchAny b1 b2
    matchAny (A a1, b1, _) (A a2, b2, Any) = matchAny a1 a2 && matchAny b1 b2
    matchAny (_, b1, A c1) (Any, b2, A c2) = matchAny b1 b2 && matchAny c1 c2

    matchAny (Any, b1, _) (_, b2, Any) = matchAny b1 b2
    matchAny (_, b1, Any) (Any, b2, _) = matchAny b1 b2

    matchAny (A a1, b1, A c1) (A a2, b2, A c2) = matchAny a1 a2 && matchAny b1 b2 && matchAny c1 c2
    matchAny _ _ = False

instance (MatchAny a, MatchAny b, MatchAny c) => MatchAny (a, Any b, Any c) where
    matchAny (a1, Any, Any) (a2, _, _) = matchAny a1 a2
    matchAny (a1, A b1, Any) (a2, A b2, _) = matchAny a1 a2 && matchAny b1 b2
    matchAny (a1, Any, A c1) (a2, _, A c2) = matchAny a1 a2 && matchAny c1 c2

    matchAny (a1, _, _) (a2, Any, Any) = matchAny a1 a2
    matchAny (a1, A b1, _) (a2, A b2, Any) = matchAny a1 a2 && matchAny b1 b2
    matchAny (a1, _, A c1) (a2, Any, A c2) = matchAny a1 a2 && matchAny c1 c2

    matchAny (a1, Any, _) (a2, _, Any) = matchAny a1 a2
    matchAny (a1, _, Any) (a2, Any, _) = matchAny a1 a2

    matchAny (a1, A b1, A c1) (a2, A b2, A c2) = matchAny a1 a2 && matchAny b1 b2 && matchAny c1 c2
    matchAny _ _ = False

-- instance (MatchAny a, MatchAny b, MatchAny c) => MatchAny (Any a, Any b, Any c) where
--     matchAny (Any, Any, Any) _ = True
--     matchAny (A a1, Any, Any) (A a2, _, _) = matchAny a1 a2
--     matchAny (Any, A b1, Any) (_, A b2, _) = matchAny b1 b2
--     matchAny (Any, Any, A c1) (_, _, A c2) = matchAny c1 c2
--     matchAny (A a1, A b1, Any) (A a2, A b2, _) = matchAny a1 a2 && matchAny b1 b2
--     matchAny (A a1, Any, A c1) (A a2, _, A c2) = matchAny a1 a2 && matchAny c1 c2
--     matchAny (Any, A b1, A c1) (_, A b2, A c2) = matchAny b1 b2 && matchAny c1 c2
--     matchAny (A a1, A b1, A c1) (A a2, A b2, A c2) = matchAny a1 a2 && matchAny b1 b2 && matchAny c1 c2
--     matchAny _ _ = False


instance (MatchAny a, MatchAny b, MatchAny c) => MatchAny (Any a, Any b, Any c) where
    matchAny (Any, Any, Any) _ = True
    matchAny (A a1, Any, Any) (A a2, _, _) = matchAny a1 a2
    matchAny (Any, A b1, Any) (_, A b2, _) = matchAny b1 b2
    matchAny (Any, Any, A c1) (_, _, A c2) = matchAny c1 c2
    matchAny (A a1, A b1, Any) (A a2, A b2, _) = matchAny a1 a2 && matchAny b1 b2
    matchAny (A a1, Any, A c1) (A a2, _, A c2) = matchAny a1 a2 && matchAny c1 c2
    matchAny (Any, A b1, A c1) (_, A b2, A c2) = matchAny b1 b2 && matchAny c1 c2
    matchAny (A a1, A b1, A c1) (A a2, A b2, A c2) = matchAny a1 a2 && matchAny b1 b2 && matchAny c1 c2

    matchAny (_, Any, Any) (Any, _, _) = True
    matchAny (Any, _, Any) (_, Any, _) = True
    matchAny (Any, Any, _) (_, _, Any) = True

    matchAny (A a1, _, Any) (A a2, Any, _) = matchAny a1 a2
    matchAny (A a1, Any, _) (A a2, _, Any) = matchAny a1 a2
    matchAny (_, A b1, Any) (Any, A b2, _) = matchAny b1 b2
    matchAny (Any, A b1, _) (_, A b2, Any) = matchAny b1 b2
    matchAny (_, Any, A c1) (Any, _, A c2) = matchAny c1 c2
    matchAny (Any, _, A c1) (_, Any, A c2) = matchAny c1 c2

    matchAny _ _ = False
