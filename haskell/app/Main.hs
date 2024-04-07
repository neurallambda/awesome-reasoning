{-

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

data FSMOutput f b s = FSMOutput {
  fsmOutput :: !(f b),
  fsmState :: !(f s)
  } deriving (Functor, Show)

-- | Construct an FSM from a transition function
fsm :: (a -> s -> FSMOutput Identity b s) -> MSF (State s) a b
fsm f = arrM (\a -> do
                 s <- gets Identity
                 let FSMOutput b s' = f a (runIdentity s)
                 put (runIdentity s')
                 pure (runIdentity b))

-- | Upgrade an MSF to also output it's state
outputState :: MSF (State s) a b -> MSF (State s) a (b, s)
outputState msf = proc a -> do
  b <- msf -< a
  s <- arrM gets -< id
  returnA -< (b, s)

-- | Run an FSM and keep a list of outputs and states
runFSM :: MSF (State s) a b -> [a] -> s -> FSMOutput [] b s
runFSM msf input initialS = let
  out = evalState (embed (outputState msf) input) initialS
  (bs, ss) = unzip out
  in FSMOutput bs ss

-- | A helper to wrap `Identity`
fsmi :: b -> s -> FSMOutput Identity b s
fsmi a b = FSMOutput (Identity a) (Identity b)

-- | Example FSM for counting a's
countA :: Char -> Int -> FSMOutput Identity () Int
countA 'a' count = fsmi () (count + 1)
countA _ count   = fsmi () count


--------------------------------------------------
-- * Push down automata

data Sym = A | B | C deriving (Eq, Show)
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
data PDATransition f b s stack = PDATransition {
  pdaTOutput :: !(f b),
  pdaTState :: !(f s),
  pdaTStack :: !(f (PDAOp stack))
  }

data PDAState state stack = PDAState {
  pdaState :: !state,
  pdaStack :: !(Seq stack)
  }

-- | Construct a PDA from a transition function
pda :: (Monad f) => (a -> s -> Maybe stack -> PDATransition Identity b s stack) -> MSF (StateT (PDAState state stack) f) a b
pda f = arrM (\a -> do
  PDAState{..} <- get
  let hd = viewl pdaStack
  let PDATransition b s' stackOp = f a (runIdentity pdaState) hd
  case runIdentity stackOp of
    NullOp -> pure ()
    Push x -> modify (x <|)
    Pop    -> modify $ Seq.drop 1
  put (runIdentity s')
  pure $ runIdentity b
               )
-- | Run a PDA and keep a list of outputs, states, and stack
runPDA :: MSF (StateT (Seq stack) Identity) a b -> [a] -> s -> Seq stack -> PDATransition [] b s stack
runPDA pdaMsf input initialS initialStack = let
  out = runIdentity $ evalStateT (embed pdaMsf input) initialStack
  (bs, ss, stacks) = unzip3 out
  in PDATransition bs ss stacks

-- | A helper to wrap `Identity`
pdai :: b -> s -> PDAOp stack -> PDATransition Identity b s stack
pdai b s op = PDATransition (Identity b) (Identity s) (Identity op)

-- | Example PDA for recognizing strings of the form a^n b^n
-- (This is a simple example and may not cover all cases)
examplePDA :: Char -> Q -> Maybe Sym -> PDATransition Identity Bool Q Sym
examplePDA 'a' Q0 Nothing  = pdai True Q1 (Push A)
examplePDA 'a' Q1 (Just A) = pdai True Q1 (Push A)
examplePDA 'b' Q1 (Just A) = pdai True Q2 Pop
examplePDA 'b' Q2 (Just A) = pdai True Q2 Pop
examplePDA 'b' Q2 Nothing  = pdai True QAccept NullOp
examplePDA _ _ _           = pdai False QReject NullOp

-- data PDAOp s =
--   NullOp
--   | Push s
--   | Pop

-- -- | Pushdown Automaton (PDA)
-- newtype PDATransition f a b state stack = PDATransition {
--   pdaOutput :: !(f b),
--   pdaState :: !(f b),
--   pdaStack :: !(f b)
--   } deriving (Show, Functor)

-- -- | Construct a PDA from a transition function
-- pda :: (a -> state -> Maybe stack -> (b, state, PDAOp stack)) -> PDATransition Identity a b state stack




-- pda f = let m = MSF $ (\a -> do
--                           (s, stk) <- get
--                           let (b, s', op) = f a s (viewl stk)
--                           case op of
--                             NullOp -> pure ()
--                             Push x -> put (s', stk |> x)
--                             Pop -> put (s', Seq.drop 1 stk)
--                           pure (b, m))
--   in PDA m

-- runPDA :: PDA a b state stack -> [a] -> state -> Seq stack -> ([b], (state, Seq stack))
-- runPDA (PDA msf) input initialState initialStack = runState (go input msf) (initialState, initialStack)
--   where
--     go [] _ = pure []
--     go (x:xs) m = do
--       (b, m') <- unMSF m x
--       bs <- go xs m'
--       pure (b:bs)


anbn :: Sym -> Q -> Maybe Sym -> ((), Q, PDAOp Sym)
anbn A Q0 Nothing = ((), Q1, Push A)
anbn A Q1 (Just A) = ((), Q1, Push A)
anbn B Q1 (Just A) = ((), Q2, Pop)
anbn B Q2 (Just A) = ((), Q2, Pop)
anbn B Q2 Nothing = ((), QAccept, NullOp)
anbn _ _ _ = ((), Q0, NullOp)


-- --------------------------------------------------
-- -- * Turing Machine


-- data TMOp s =
--   EnqL s
--   | DeqR
--   | TMNullOp
--   deriving (Eq, Show)

-- -- | Turing Machine with Queue
-- newtype TM a b state symbol = TM {unTM :: MSF (State (state, Seq symbol)) a b}

-- -- | Construct a TM from a transition function
-- tm :: (a -> state -> Maybe symbol -> (b, state, TMOp symbol)) -> TM a b state symbol
-- tm f = let m = MSF $ (\a -> do
--                         (s, q) <- get
--                         let (b, s', op) = f a s (viewl q)
--                         case op of
--                           EnqL new -> put (s', (Seq.<| q) new)
--                           DeqR -> put (s', dropr 1 q)
--                           TMNullOp -> pure ()
--                         pure (b, m))
--   in TM m

-- runTM :: TM a b state symbol -> [a] -> state -> Seq symbol -> ([b], (state, Seq symbol))
-- runTM (TM msf) input initialState initialQueue = runState (go input msf) (initialState, initialQueue)
--   where
--     go [] _ = pure []
--     go (x:xs) m = do
--       (b, m') <- unMSF m x
--       bs <- go xs m'
--       pure (b:bs)


-- anbncn :: TM Sym () Q Sym
-- anbncn = tm transition
--   where
--     transition :: Sym -> Q -> Maybe Sym -> ((), Q, TMOp Sym)
--     transition A Q0 _ = ((), Q0, EnqL A)
--     transition B Q0 (Just A) = ((), Q1, EnqL A)


--------------------------------------------------
-- Go

main :: IO ()
main = do
  -- FSM
  putStrLn "----------"
  putStrLn "-- FSM"
  let str = "aaabaa"
  print $ runFSM (fsm countA) str 0


  -- -- PDA
  -- putStrLn "----------"
  -- putStrLn "-- PDA"
  -- let strings = [ [A, A, B, B],
  --                 [A, A, B, B, B],
  --                 [A, A, A, B, B]
  --               ]
  -- mapM_ (\str -> do
  --   let (out, (state, stack)) = runPDA anbn str Q0 Seq.empty
  --   putStrLn $ show str
  --     ++ " : " ++ show out
  --     ++ " : " ++ show state
  --     ++ " : " ++ show stack) strings


  -- -- TM
  -- putStrLn "----------"
  -- putStrLn "-- Turing Machine"
  -- let strings = [
  --       [A],
  --       [A, B],
  --       [A, B, C],
  --       [A, B, C, C],
  --       [A, B, B, C],
  --       [A, A, B, B, C, C],
  --       [A, A, B, B, B, C, C],
  --       [A, A, A, B, B, C, C, C]
  --       ]
  -- mapM_ (\str -> do
  --   let (out, (state, queue)) = runTM anbncn str Q0 Seq.empty
  --   putStrLn $ show str
  --     ++ " : " ++ show out
  --     ++ " : " ++ show state
  --     ++ " : " ++ show queue) strings
