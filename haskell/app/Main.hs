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

module Main where

import Data.Sequence (Seq, ViewL(..), (<|))
import qualified Data.Sequence as Seq
import Data.Foldable ( toList, foldl', find )
import Data.List (intercalate, nub)
import qualified Data.Map as M
import Data.Kind (Type)
import Data.Map.Extra (lookupMatchAny, Any(..), MatchAny(..))
import Control.Monad (guard)
import qualified Data.Sequence as Seq
import Data.Sequence (Seq(..), (|>))
import Data.Maybe (mapMaybe)
import Debug.Trace
import Data.Sequence (Seq(..), (|>))

-- -- | Generate valid strings of the Left portion of the transition relation.
-- generateLs :: (Machine m a s, Ord (L m a s), MatchAny (L m a s))
--   => [(L m a s, R m a s)] -- transition relation
--   -> (S m a s -> Bool) -- halting function TODO: this could eventually be a MatchAnyable thing
--   -> [a] -- input symbols
--   -> S m a s -- initial state
--   -> [[L m a s]] -- lazy list of valid prefixes
-- generateLs transitions halt syms initialState = go initialState []
--   where
--     -- go :: S m a s
--     go state acc
--       | halt state = [acc]
--       | otherwise = do
--           a <- syms
--           let validTransitions = filter (\(l, _) -> matchAny l (mkL a state)) transitions
--           (l, r) <- validTransitions
--           let newState = action r state
--           if halt newState
--             then [acc]
--             else go newState (l : acc)

generateLs :: forall m a s. (Machine m a s, Ord (L m a s), MatchAny (L m a s))
  => [(L m a s, R m a s)] -- transition relation
  -> (S m a s -> Bool) -- halting function
  -> [a] -- input symbols
  -> S m a s -- initial state
  -> [[L m a s]] -- lazy list of valid prefixes
generateLs transitions halt syms initialState = bfs [(initialState, Empty)]
  where
    bfs :: [(S m a s, Seq (L m a s))] -> [[L m a s]]
    bfs [] = []
    bfs ((state, acc) : queue) = let
          -- find all valid transitions for each input symbol
          validTransitions = concatMap (\a ->
            let ls = filter (\(l, _) -> matchAny l (mkL a state)) transitions
            in map (\(l, r) -> (action r state, acc |> l)) ls
            ) syms
          -- add new states and their accumulators to the queue
          newQueue = queue ++ validTransitions
          -- check if the current state is a halting state
          haltingAccs = if halt state then [toList acc] else []
        in
          haltingAccs ++ bfs newQueue

-- generateLs :: forall m a s. (Machine m a s, Ord (L m a s), MatchAny (L m a s))
--   => [(L m a s, R m a s)] -- transition relation
--   -> (S m a s -> Bool) -- halting function
--   -> [a] -- input symbols
--   -> S m a s -- initial state
--   -> [[L m a s]] -- lazy list of valid prefixes
-- generateLs transitions halt syms initialState = bfs [(initialState, Empty)]
--   where
--     bfs :: [(S m a s, Seq (L m a s))] -> [[L m a s]]
--     bfs [] = []
--     bfs ((state, acc) : queue) = let
--           -- find all valid transitions for each input symbol
--           validTransitions = concatMap (\a ->
--             let ls = filter (\(l, _) -> matchAny l (mkL a state)) transitions
--             in map (\(l, r) -> (action r state, acc |> l)) ls
--             ) syms
--           -- add new states and their accumulators to the queue
--           newQueue = queue ++ validTransitions
--         in
--           -- if any new state is a halting state, convert its accumulator to list and return
--           -- otherwise, recursively search the updated queue
--           case find (halt . fst) validTransitions of
--             Just (_, acc') -> [toList acc']
--             Nothing -> toList acc : bfs newQueue

-- generateLs :: forall m a s. (Machine m a s, Ord (L m a s), MatchAny (L m a s))
--   => [(L m a s, R m a s)] -- transition relation
--   -> (S m a s -> Bool) -- halting function
--   -> [a] -- input symbols
--   -> S m a s -- initial state
--   -> [[L m a s]] -- lazy list of valid prefixes
-- generateLs transitions halt syms initialState = bfs [(initialState, Empty)]
--   where
--     bfs :: [(S m a s, Seq (L m a s))] -> [[L m a s]]
--     bfs [] = []
--     bfs ((state, acc) : queue)
--       | halt state = [toList acc]
--       | otherwise = let
--           -- find all valid transitions for each input symbol
--           validTransitions = concatMap (\a ->
--             let ls = filter (\(l, _) -> matchAny l (mkL a state)) transitions
--             in map (\(l, r) -> (action r state, acc |> l)) ls
--             ) syms
--           -- add new states and their accumulators to the queue
--           newQueue = queue ++ validTransitions
--         in
--           toList acc : bfs newQueue

-- generateLs :: forall m a s. (Machine m a s, Ord (L m a s), MatchAny (L m a s))
--   => [(L m a s, R m a s)] -- transition relation
--   -> (S m a s -> Bool) -- halting function
--   -> [a] -- input symbols
--   -> S m a s -- initial state
--   -> [[L m a s]] -- lazy list of valid prefixes
-- generateLs transitions halt syms initialState = filter (halt . fst) (bfs [(initialState, Empty)])
--   where
--     bfs :: [(S m a s, Seq (L m a s))] -> [(S m a s, Seq (L m a s))]
--     bfs [] = []
--     bfs ((state, acc) : queue) = (state, acc) : let
--           -- find all valid transitions for each input symbol
--           validTransitions = concatMap (\a ->
--             let ls = filter (\(l, _) -> matchAny l (mkL a state)) transitions
--             in map (\(l, r) -> (action r state, acc |> l)) ls
--             ) syms
--           -- add new states and their accumulators to the queue
--           newQueue = queue ++ validTransitions
--         in
--           bfs newQueue

pdaString :: (Eq a
             , Ord a
             , Ord st
             , Ord sk
             , Show a, Show st, Show sk
             )
  => [(L PDA a (st, sk), R PDA a (st, sk))]
  -> (S PDA a (st, sk) -> Bool) -- halting states
  -> [a] -- input symbols (to possibly stand in for Any)
  -> S PDA a (st, sk) -- initial state
  -> [[a]]
pdaString transitions haltStates syms initialState = mapMaybe (mapM f) (generateLs transitions haltStates syms initialState)
  where
    f (PDAL (A a) _ _) = Just a
    f (PDAL Any _ _) = Nothing


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
  data L m a s -- ^ the Left side of a delta function/relation
  data R m a s -- ^ the Right side of a delta function/relation
  data S m a s -- ^ the State of the Machine
  -- | update the state (ex apply stack ops)
  action :: R m a s -> S m a s -> S m a s
  -- | build an input (ex add a peek at the top of a stack)
  mkL :: a -> S m a s -> L m a s

-- | Run a machine on an input symbol
runStep :: (Machine m a s, Ord (L m a s), Show (L m a s), MatchAny (L m a s))
  => M.Map (L m a s) (R m a s) -- transition table
  -> S m a s -- state
  -> a -- single input
  -> (R m a s, S m a s) -- (transition value, new state)
runStep table st input =
  case lookupMatchAny (mkL input st) table of
    Just transition -> (transition, action transition st)
    Nothing -> error $ "transition not found: " ++ show (mkL input st)

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

data PDA

data PDAOp stack = NullOp | Push !stack | Pop deriving (Eq, Ord, Show)

--instance Machine PDA a (state, stack) where
instance Machine PDA a (state, stack) where
  data L PDA a (state, stack) = PDAL (Any a) (Any state) (Maybe (Any stack)) deriving (Show, Ord, Eq)
  data R PDA a (state, stack) = PDAR state (PDAOp stack) deriving (Show, Ord, Eq)
  data S PDA a (state, stack) = PDAS state (Data.Sequence.Seq stack) deriving (Show, Ord, Eq)

  action (PDAR newState NullOp) (PDAS _ stack) = PDAS newState stack
  action (PDAR newState (Push x)) (PDAS _ stack) = PDAS newState (x Data.Sequence.<| stack)
  action (PDAR newState Pop) (PDAS _ stack) = PDAS newState (Seq.drop 1 stack)

  mkL a (PDAS st sk) = PDAL (A a) (A st) (A <$> viewl sk)


instance (Eq state, Eq stack, Eq a) => MatchAny (L PDA a (state, stack)) where
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

instance MatchAny Q where matchAny x y = x == y


-- | Grammar for: a^nb^n
anbnTransitions :: [(L PDA Char (Q, Char), R PDA a (Q, Char))]
anbnTransitions = [
    (PDAL (A '^') (A Q0) Nothing, PDAR Q1 (Push '$')),
    (PDAL (A '$') (A Q1) (Just $ A '$'), PDAR QAccept Pop),
    (PDAL (A 'a') (A Q1) (Just $ A '$'), PDAR Q1 (Push 'A')),
    (PDAL (A 'a') (A Q1) (Just $ A 'A'), PDAR Q1 (Push 'A')),
    (PDAL (A 'b') (A Q1) (Just $ A 'A'), PDAR Q2 Pop),
    (PDAL (A 'b') (A Q2) (Just $ A 'A'), PDAR Q2 Pop),
    (PDAL (A '$') (A Q2) (Just $ A '$'), PDAR QAccept Pop)

    -- NOTE: These only work during parsing ie when transitions are ordered. For
    --       string generation they don't really work.
    --
    -- (PDAL Any Any Nothing, PDAR QReject NullOp),
    -- (PDAL Any Any (Just Any), PDAR QReject NullOp)
  ]


----------
-- * Go

showSeq :: Show a => Data.Sequence.Seq a -> String
showSeq xs = "{" ++ intercalate ", " (show <$> toList xs) ++ "}"

formatPDAResult :: (R PDA Char (Q, Char), S PDA Char (Q, Char)) -> String
formatPDAResult (_, PDAS finalState stack) =
  "Final state: " ++ show finalState ++ " | Stack: " ++ showSeq stack

halt :: S PDA a (Q, stack) -> Bool
halt (PDAS QReject _) = True
halt (PDAS QAccept _) = True
halt _ = False

main :: IO ()
main = do

  -- -- PDA
  -- putStrLn "----------"
  -- putStrLn "-- PDA"
  -- let trans = M.fromList anbnTransitions
  --     exampleStrings = [
  --       "^aaabbb$",
  --       "^aabbb$",
  --       "^aaabb$",
  --       "^ab$",
  --       "^$"]
  -- putStrLn "PDA Example strings:"
  -- mapM_ (putStrLn . formatPDAResult . runMachine trans (PDAS Q0 Seq.empty)) exampleStrings

  -- -- Generate valid PDA strings

  let -- trans = M.fromList anbnTransitions
      initialState = PDAS Q0 Seq.empty
      inputSymbols = ['^', 'a', 'b', '$']
      haltStates = [PDAS QReject Seq.empty] -- need Any, and need Functor instance for state
  let strings = pdaString anbnTransitions halt inputSymbols initialState

  putStrLn "generations:"
  mapM_ print (take 10 strings)
  putStrLn "done:"
