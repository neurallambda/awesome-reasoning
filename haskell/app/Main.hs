{-

Generate sentences in given grammars.


----------
NOTES:

Mealy Machine (FSM): 6-tuple (Q, Σ, Δ, δ, λ, q₀), where:
- Q is a finite set of states
- Σ is a finite set of input symbols (the input alphabet)
- Δ is a finite set of output symbols (the output alphabet)
- δ : Q × Σ → Q is the state transition function
- λ : Q × Σ → Δ is the output function
- q₀ ∈ Q is the initial state

Moore Machine (FSM): 6-tuple (Q, Σ, Δ, δ, λ, q₀), where:
- Q is a finite set of states
- Σ is a finite set of input symbols (the input alphabet)
- Δ is a finite set of output symbols (the output alphabet)
- δ : Q × Σ → Q is the state transition function
- λ : Q → Δ is the output function
- q₀ ∈ Q is the initial state

Pushdown Automaton (PDA): 7-tuple (Q, Σ, Γ, δ, q₀, Z₀, F), where:
- Q is a finite set of states
- Σ is a finite set of input symbols (the input alphabet)
- Γ is a finite set of stack symbols (the stack alphabet)
- δ : Q × (Σ ∪ {ε}) × Γ → P(Q × Γ*) is the transition function
- q₀ ∈ Q is the initial state
- Z₀ ∈ Γ is the initial stack symbol
- F ⊆ Q is the set of accepting/final states

Turing Machine (TM) with Queue: 8-tuple (Q, Σ, Γ, δ, q₀, q_acc, q_rej, □), where:
- Q is a finite set of states
- Σ is a finite set of input symbols (the input alphabet)
- Γ is a finite set of queue symbols (the queue alphabet), where Σ ⊆ Γ and □ ∈ Γ \ Σ (□ is the blank symbol)
- δ : Q × Γ → Q × (Γ ∪ {ε}) × {EnqL, EnqR, DeqL, DeqR} is the transition function
- q₀ ∈ Q is the initial state
- q_acc ∈ Q is the accepting state
- q_rej ∈ Q is the rejecting state
- □ is the blank symbol

-}

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE Arrows #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad.Identity
import Control.Monad.State
import Data.Sequence (Seq, ViewL(..), (<|), (|>))
import qualified Data.Sequence as Seq
import Control.Arrow
import Data.MonadicStreamFunction
import Data.Foldable (toList)
import Data.List (intercalate)
import qualified Data.Map as M
import System.Random (randomRIO)


--------------------------------------------------
-- * Util

viewl :: Seq.Seq a -> Maybe a
viewl s = case Seq.viewl s of
  Seq.EmptyL -> Nothing
  (x Seq.:< _) -> Just x

viewr :: Seq.Seq a -> Maybe a
viewr s = case Seq.viewr s of
  Seq.EmptyR -> Nothing
  (_ Seq.:> x) -> Just x

splitAtR :: Int -> Seq.Seq a -> (Seq.Seq a, Seq.Seq a)
splitAtR i s = Seq.splitAt (length s - i) s

taker :: Int -> Seq.Seq a -> Seq.Seq a
taker i s = snd $ splitAtR i s

dropr :: Int -> Seq.Seq a -> Seq.Seq a
dropr i s = fst $ splitAtR i s


--------------------------------------------------
-- * Finite State Machine

data FSMTransition out state = FSMTransition {
  fsmtOut :: !out,
  fsmNewState :: !state
  } deriving Show

-- | Construct an FSM from a transition function
fsm :: (a -> s -> FSMTransition b s) -> MSF (State s) a b
fsm f = arrM (\a -> do
                 s <- get
                 let FSMTransition b s' = f a s
                 put s'
                 pure b)

-- | Upgrade an MSF to also output it's state
outputState :: MSF (State s) a b -> MSF (State s) a (b, s)
outputState msf = proc a -> do
  b <- msf -< a
  s <- arrM gets -< id
  returnA -< (b, s)

-- | Run an FSM and keep a list of outputs and states
runFSM :: MSF (State s) a b -> [a] -> s -> ([b], [s])
runFSM msf input initialS = let
  out = evalState (embed (outputState msf) input) initialS
  in unzip out

-- | Run an FSM to it's final output and state
runFSM' :: MSF (State s) a b -> [a] -> s -> (b, s)
runFSM' msf input initialS = let
  out = evalState (embed (outputState msf) input) initialS
  in (fst (last out), snd (last out))

-- | Evaluate an FSM and collect its outputs (not state)
evalFSM :: MSF (State s) a b -> [a] -> s -> [b]
evalFSM msf input initialS = let
  out = evalState (embed msf input) initialS
  in out

-- | Evaluate an FSM to its final output
evalFSM' :: MSF (State s) a b -> [a] -> s -> b
evalFSM' msf input initialS = let
  out = evalState (embed msf input) initialS
  in last out

-- | Evaluate an FSM and collect its states (not outputs)
execFSM :: MSF (State s) a b -> [a] -> s -> [s]
execFSM msf input initialS = let
  msf' = proc a -> do
    _ <- msf -< a
    arrM gets -< id
  out = evalState (embed msf' input) initialS
  in out

-- | Evaluate an FSM to its final state
execFSM' :: MSF (State s) a b -> [a] -> s -> s
execFSM' msf input initialS = let
  msf' = proc a -> do
    _ <- msf -< a
    arrM gets -< id
  out = evalState (embed msf' input) initialS
  in last out


-- | Example FSM for counting a's
countA :: Char -> Int -> FSMTransition () Int
countA 'a' count = FSMTransition () (count + 1)
countA _ count   = FSMTransition () count


--------------------------------------------------
-- * Push down automata


-- * Push down automata

data PDAOp s =
  NullOp
  | Push !s
  | Pop
  deriving (Eq, Show)

-- | Pushdown Automaton (PDA)
data PDATransition b state stack = PDATransition {
  pdaOutput :: !b,
  pdaState :: !state,
  pdaStack :: !(PDAOp stack)
  }

type PDAState state stack = (state, Seq stack)
type TransitionTable state stack = M.Map (state, Maybe stack) (M.Map Char (PDATransition () state stack))

-- | Run a PDA on an input string
runPDA :: (Ord state, Ord stack) => TransitionTable state stack -> [Char] -> state -> (Maybe state, Seq stack)
runPDA table input initialState = go input initialState Seq.empty
  where
    go [] state stack = (Just state, stack)
    go (c:cs) state stack =
      case M.lookup (state, viewl stack) table >>= M.lookup c of
        Just (PDATransition _ nextState op) ->
          let stack' = case op of
                         NullOp -> stack
                         Push x -> x <| stack
                         Pop    -> Seq.drop 1 stack
          in go cs nextState stack'
        Nothing -> (Nothing, stack)



-- | Generate a valid string from a PDA
generateString :: (Ord state, Ord stack) => TransitionTable state stack -> state -> Seq stack -> Int -> IO (Maybe String)
generateString table state stack maxDepth = go state stack [] 0
  where
    go state stack str depth
      | depth > maxDepth = pure $ Just (reverse str)
      | otherwise = do
          let transitions = M.findWithDefault M.empty (state, viewl stack) table
          if M.null transitions
            then pure $ Just (reverse str)
            else do
              let inputs = M.keys transitions
              i <- randomRIO (0, length inputs - 1)
              let c = inputs !! i
                  PDATransition _ nextState op = transitions M.! c
                  stack' = case op of
                             NullOp -> stack
                             Push x -> x <| stack
                             Pop    -> Seq.drop 1 stack
              go nextState stack' (c:str) (depth + 1)


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

buildTransitionTable :: TransitionTable Q Char
buildTransitionTable = M.fromList [
    ((Q0, Nothing),      M.fromList [('^', PDATransition () Q1 (Push '$'))]),
    ((Q1, Just '$'),     M.fromList [('$', PDATransition () QAccept Pop),
                                     ('a', PDATransition () Q1 (Push 'A'))]),
    ((Q1, Just 'A'),     M.fromList [('a', PDATransition () Q1 (Push 'A')),
                                     ('b', PDATransition () Q2 Pop)]),
    ((Q2, Just 'A'),     M.fromList [('b', PDATransition () Q2 Pop)]),
    ((Q2, Just '$'),     M.fromList [('$', PDATransition () QAccept Pop)])
  ]


----------
-- * Go

showSeq :: Show a => Seq a -> String
showSeq xs = "{" ++ intercalate ", " (show <$> toList xs) ++ "}"

formatPDAResult :: (Maybe Q, Seq Char) -> String
formatPDAResult (finalState, stack) =
  "Final state: " ++ show finalState ++ " | Stack: " ++ showSeq stack

main :: IO ()
main = do
  -- PDA
  putStrLn "----------"
  putStrLn "-- PDA"
  let transitionTable = buildTransitionTable
      exampleStrings = [
        "^aaabbb$",
        "^aabbb$",
        "^aaabb$",
        "^ab$",
        "^$"]
  putStrLn "Example strings:"
  mapM_ (putStrLn . formatPDAResult . (\x -> runPDA transitionTable x Q0)) exampleStrings

  -- Generate valid strings
  putStrLn "----------"
  putStrLn "-- Generate valid strings"
  replicateM_ 5 $ do
    result <- generateString transitionTable Q0 Seq.empty 100
    case result of
      Just str -> putStrLn str
      Nothing  -> putStrLn "Failed to generate a valid string"
