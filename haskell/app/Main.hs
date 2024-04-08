{-

Generate sentences in given grammars.


----------
NOTES:

Mealy Machine (FSM): 6-tuple (Q, Σ, Δ, δ, λ, q₀), where:
- Q : states
- Σ : input symbols
- Δ : output symbols
- δ : Q × Σ → Q is the state transition function
- λ : Q × Σ → Δ is the output function
- q₀ ∈ Q is the initial state

Moore Machine (FSM): 6-tuple (Q, Σ, Δ, δ, λ, q₀), where:
- Q : states
- Σ : input symbols
- Δ : output symbols
- δ : Q × Σ → Q is the state transition function
- λ : Q → Δ is the output function
- q₀ ∈ Q

Pushdown Automaton (PDA): 7-tuple (Q, Σ, Γ, δ, q₀, Z₀, F), where:
- Q : states
- Σ : input symbols
- Γ : stack symbols (the stack alphabet)
- δ : Q × (Σ ∪ {ε}) × Γ → P(Q × Γ*) is the transition function
- q₀ ∈ Q
- Z₀ ∈ Γ is the initial stack symbol
- F ⊆ Q is the set of accepting/final states

Turing Machine (TM) with Queue: 8-tuple (Q, Σ, Γ, δ, q₀, q_acc, q_rej, □), where:
- Q : states
- Σ : input symbols
- Γ : queue symbols (the queue alphabet), where Σ ⊆ Γ and □ ∈ Γ \ Σ (□ is the blank symbol)
- δ : Q × Γ → Q × (Γ ∪ {ε}) × {EnqL, EnqR, DeqL, DeqR} is the transition function
- q₀ ∈ Q
- q_acc ∈ Q is the accepting state
- q_rej ∈ Q is the rejecting state
- □ is the blank symbol

Queue Automaton, Γ is head, Γ* is queue:
δ : Q × Γ → Q × Γ*

Multi-tape, with k tapes:
δ : Q × Γ_k → Q × Γ_k × { L , R }

-}

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
-- {-# LANGUAGE OverlappingInstances #-}

module Main where

import Data.Sequence (Seq, ViewL(..), (<|))
import qualified Data.Sequence as Seq
import Data.Foldable (toList)
import Data.List (intercalate)
import qualified Data.Map as M
import Data.Kind (Type)
import Data.Foldable (foldl')
import Data.Map.Extra (lookupMatchAny, Any(..), MatchAny(..))


--------------------------------------------------
-- * Util

viewl :: Data.Sequence.Seq a -> Maybe a
viewl s = case Seq.viewl s of
  Data.Sequence.EmptyL -> Nothing
  (x Data.Sequence.:< _) -> Just x

viewr :: Data.Sequence.Seq a -> Maybe a
viewr s = case Seq.viewr s of
  Seq.EmptyR -> Nothing
  (_ Seq.:> x) -> Just x

splitAtR :: Int -> Data.Sequence.Seq a -> (Data.Sequence.Seq a, Data.Sequence.Seq a)
splitAtR i s = Seq.splitAt (length s - i) s

taker :: Int -> Data.Sequence.Seq a -> Data.Sequence.Seq a
taker i s = snd $ splitAtR i s

dropr :: Int -> Data.Sequence.Seq a -> Data.Sequence.Seq a
dropr i s = fst $ splitAtR i s

--------------------------------------------------
-- * Machine Definition

class Machine m a (s :: Type) where
  data L m a s
  data R m a s
  data S m a s
  action :: R m a s -> S m a s -> S m a s
  mkInput :: a -> S m a s -> L m a s


-- -- | Run a machine on an input symbol
-- execStep :: (Machine m a s, Ord (L m a s), Show (L m a s))
--   => M.Map (L m a s) (R m a s) -- transition table
--   -> a -- single input
--   -> S m a s -- state
--   -> S m a s -- new state
-- execStep table input st =
--   case M.lookup (mkInput input st) table of
--     Just transition -> action transition st
--     Nothing -> error $ "transition not found: " ++ show (mkInput input st)


-- -- | Run a machine on a list of input symbols
-- execMachine :: (Machine m a s, Ord (L m a s), Show (L m a s))
--   => M.Map (L m a s) (R m a s) -- transition table
--   -> [a] -- input symbols
--   -> S m a s -- initial state
--   -> S m a s
-- execMachine table input initialState = foldr (execStep table) initialState input


-- | Run a machine on an input symbol
runStep :: (Machine m a s, Ord (L m a s), Show (L m a s), MatchAny (L m a s))
  => M.Map (L m a s) (R m a s) -- transition table
  -> S m a s -- state
  -> a -- single input
  -> (R m a s, S m a s) -- (transition value, new state)
runStep table st input =
  case lookupMatchAny (mkInput input st) table of
    Just transition -> (transition, action transition st)
    Nothing -> error $ "transition not found: " ++ show (mkInput input st)

-- | Run a machine on a list of input symbols
runMachine :: (Machine m a s
              , Ord (L m a s)
              , Show (L m a s)
              , MatchAny (L m a s)
              )
  => M.Map (L m a s) (R m a s) -- transition table
  -> S m a s -- initial state
  -> [a] -- input symbols
  -> (R m a s, S m a s)
runMachine table initialState = foldl' f (error "empty input", initialState)
  where f (_, state) = runStep table state


--------------------------------------------------
-- * Push down automata

data PDA s

data PDAOp stack = NullOp | Push stack | Pop deriving (Eq, Ord, Show)

instance Machine PDA a (state, stack) where
  data L PDA a (state, stack) = PDAL (Any a) (Any state) (Maybe (Any stack)) deriving (Show, Ord, Eq)
  data R PDA a (state, stack) = PDAR state (PDAOp stack) deriving (Show, Ord, Eq)
  data S PDA a (state, stack) = PDAS state (Data.Sequence.Seq stack) deriving (Show, Ord, Eq)

  action (PDAR newState NullOp) (PDAS _ stack) = PDAS newState stack
  action (PDAR newState (Push x)) (PDAS _ stack) = PDAS newState (x Data.Sequence.<| stack)
  action (PDAR newState Pop) (PDAS _ stack) = PDAS newState (Seq.drop 1 stack)

  mkInput a (PDAS st sk) = PDAL (A a) (A st) (A <$> (viewl sk))


-- needs OverlappingInstances
instance (MatchAny a, MatchAny state, Eq stack, Eq a, MatchAny Char) => MatchAny (L PDA a (state, stack)) where
  matchAny (PDAL x0 x1 x2) (PDAL y0 y1 y2) = matchAny (x0, x1, x2) (y0, y1, y2)


----------
-- * Example

data Q =
  Q0
  | Q1
  | Q2
  | Q3
  | Q4
  | QAccept
  | QReject
  deriving (Eq, Show, Ord)


-- TODO: this is messy
instance MatchAny Q where
  matchAny x y = x == y

instance MatchAny Char where
  matchAny x y = x == y


-- | Grammar for: a^nb^n
anbnTransitions :: [(L PDA Char (Q, Char), R PDA a (Q, Char))]
anbnTransitions = [
    (PDAL (A '^') (A Q0) Nothing,    PDAR Q1 (Push '$')),
    (PDAL (A '$') (A Q1) (Just $ A '$'), PDAR QAccept Pop),
    (PDAL (A 'a') (A Q1) (Just $ A '$'), PDAR Q1 (Push 'A')),
    (PDAL (A 'a') (A Q1) (Just $ A 'A'), PDAR Q1 (Push 'A')),
    (PDAL (A 'b') (A Q1) (Just $ A 'A'), PDAR Q2 Pop),
    (PDAL (A 'b') (A Q2) (Just $ A 'A'), PDAR Q2 Pop),
    (PDAL (A '$') (A Q2) (Just $ A '$'), PDAR QAccept Pop),

    (PDAL Any Any Nothing, PDAR QReject NullOp),
    (PDAL Any Any (Just Any), PDAR QReject NullOp)
  ]


----------
-- * Go

showSeq :: Show a => Data.Sequence.Seq a -> String
showSeq xs = "{" ++ intercalate ", " (show <$> toList xs) ++ "}"

--formatPDAResult :: (R PDA Char (Q, Char), S PDA Char (Q, Char)) -> String
formatPDAResult (_, PDAS finalState stack) =
  "Final state: " ++ show finalState ++ " | Stack: " ++ showSeq stack

main :: IO ()
main = do

  -- PDA
  putStrLn "----------"
  putStrLn "-- PDA"
  let trans = M.fromList anbnTransitions
      exampleStrings = [
        "^aaabbb$",
        "^aabbb$",
        "^aaabb$",
        "^ab$",
        "^$"]
  putStrLn "PDA Example strings:"
  mapM_ (putStrLn . formatPDAResult . (\x -> runMachine trans (PDAS Q0 Seq.empty) x)) exampleStrings

  -- -- Generate valid PDA strings
  -- putStrLn "----------"
  -- putStrLn "-- Generate valid PDA strings"
  -- replicateM_ 5 $ do
  --   result <- generateString transitionTable Q0 Seq.empty 100
  --   case result of
  --     Just str -> putStrLn str
  --     Nothing  -> putStrLn "Failed to generate a valid PDA string"
