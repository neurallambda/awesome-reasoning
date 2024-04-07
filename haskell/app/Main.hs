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

data Q =
  Q0
  | Q1
  | Q2
  | Q3
  | Q4
  | QAccept
  | QReject
  deriving (Eq, Show)

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

-- | Construct an FSM from a transition function
pda ::
  (a -> state -> Maybe stack -> PDATransition b state stack)
  -> MSF (State (PDAState state stack)) a b
pda f = arrM (\a -> do
                 (st, sk) <- get
                 let PDATransition b st' skOp = f a st (viewl sk)
                 case skOp of
                   NullOp -> put (st', sk)
                   Push x -> put (st', x <| sk)
                   Pop -> put (st', Seq.drop 1 sk)

                 pure b)

-- | Upgrade an MSF to also output it's state
outputState' :: MSF (State s) a b -> MSF (State s) a (b, s)
outputState' msf = proc a -> do
  b <- msf -< a
  s <- arrM gets -< id
  returnA -< (b, s)

-- | Run an FSM and keep a list of outputs and states
runPDA :: MSF (State (PDAState state stack)) a b -> [a] -> state -> ([b], [state], [Seq stack])
runPDA msf input initialSt = let
  msf' = proc a -> do
    b <- msf -< a
    (st, sk) <- arrM gets -< id
    returnA -< (b, st, sk)
  out = evalState (embed msf' input) (initialSt, Seq.empty)
  in unzip3 out

anbn ::
  Char
  -> Q    -- state
  -> Maybe Char -- stack
  -> PDATransition () Q Char
-- If the input is '^' and the state is Q0 with an empty stack, push '$' onto the stack and move to Q1
anbn '^' Q0 Nothing      = PDATransition () Q1 (Push '$')
-- If the input is '$' and the state is Q1 with '$' on the stack, move to QAccept and pop '$'
anbn '$' Q1 (Just '$')   = PDATransition () QAccept Pop
-- If the input is 'a' and the state is Q1 with '$' on the stack, push 'A' onto the stack and stay in Q1
anbn 'a' Q1 (Just '$')   = PDATransition () Q1 (Push 'A')
-- If the input is 'a' and the state is Q1 with 'A' on the stack, push 'A' onto the stack and stay in Q1
anbn 'a' Q1 (Just 'A')   = PDATransition () Q1 (Push 'A')
-- If the input is 'b' and the state is Q1 with 'A' on the stack, pop 'A' from the stack and move to Q2
anbn 'b' Q1 (Just 'A')   = PDATransition () Q2 Pop
-- If the input is 'b' and the state is Q2 with 'A' on the stack, pop 'A' from the stack and stay in Q2
anbn 'b' Q2 (Just 'A')   = PDATransition () Q2 Pop
-- If the input is '$' and the state is Q2 with '$' on the stack, move to QAccept and pop '$'
anbn '$' Q2 (Just '$')   = PDATransition () QAccept Pop
-- If the input is '$' and the state is Q2 with 'A' on the stack, move to QReject
anbn '$' Q2 (Just 'A')   = PDATransition () QReject NullOp
-- For any other input or state configuration, move to QReject
anbn _ _ _               = PDATransition () QReject NullOp



----------
-- * Go

showSeq :: Show a => Seq a -> String
showSeq xs = "{" ++ intercalate ", " (show <$> toList xs) ++ "}"

formatPDAResult :: ([()], [Q], [Seq Char]) -> String
formatPDAResult (outs, states, stack) =
  let finalState = last states
  in "Final state: " ++ show finalState ++ " | " ++ show (showSeq <$> stack)

main :: IO ()
main = do
  -- FSM
  putStrLn "----------"
  putStrLn "-- FSM"
  let str = "aaabaa"
  print $ runFSM (fsm countA) str 0

  -- PDA
  putStrLn "----------"
  putStrLn "-- PDA"
  let exampleStrings = [
        "^aaabbb$",
        "^aabbb$",
        "^aaabb$",
        "^ab$",
        "^$"]
  putStrLn "Example strings:"
  mapM_ (putStrLn . formatPDAResult . (\x -> runPDA (pda anbn) x Q0)) exampleStrings
