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

module Main where

import Control.Monad.Identity
import Control.Monad.State
import Data.Sequence (Seq, ViewL(..), (<|), (|>))
import qualified Data.Sequence as Seq
import Control.Arrow (second)
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
-- * Monadic Stream Function

-- | Monadic State Function
data MSF m a b = MSF { unMSF :: a -> m (b, MSF m a b) }


--------------------------------------------------
-- * Finite State Machine

-- Finite State Machine (FSM)
newtype FSM a b s = FSM {unFSM :: MSF (State s) a b}

-- Construct an FSM from a transition function
fsm :: (a -> s -> (b, s)) -> FSM a b s
fsm f = let m = MSF $ (\a -> do
                          s <- get
                          let (b, s') = f a s
                          put s'
                          pure (b, m))
  in FSM m

runFSM :: FSM a b s -> [a] -> s -> ([b], s)
runFSM (FSM msf) input s = runState (go input msf) s
  where
    go [] _ = pure []
    go (x:xs) m = do
      (b, m') <- unMSF m x
      bs <- go xs m'
      pure (b:bs)


-- Example FSM for counting a's
countA :: FSM Char () Int
countA = fsm transition
  where
    transition 'a' count = ((), count + 1)
    transition _ count = ((), count)


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

data Op s =
  NullOp
  | Push s
  | Pop

-- | Pushdown Automaton (PDA)
newtype PDA a b state stack = PDA {unPDA :: MSF (State (state, Seq stack)) a b}

-- | Construct a PDA from a transition function
pda :: (a -> state -> Maybe stack -> (b, state, Op stack)) -> PDA a b state stack
pda f = let m = MSF $ (\a -> do
                          (s, stk) <- get
                          let (b, s', op) = f a s (viewl stk)
                          case op of
                            NullOp -> pure ()
                            Push x -> put (s', stk |> x)
                            Pop -> put (s', Seq.drop 1 stk)
                          pure (b, m))
  in PDA m

runPDA :: PDA a b state stack -> [a] -> state -> Seq stack -> ([b], (state, Seq stack))
runPDA (PDA msf) input initialState initialStack = runState (go input msf) (initialState, initialStack)
  where
    go [] _ = pure []
    go (x:xs) m = do
      (b, m') <- unMSF m x
      bs <- go xs m'
      pure (b:bs)

anbn :: PDA Sym () Q Sym
anbn = pda transition
  where
    transition :: Sym -> Q -> Maybe Sym -> ((), Q, Op Sym)
    transition A Q0 Nothing = ((), Q1, Push A)
    transition A Q1 (Just A) = ((), Q1, Push A)
    transition B Q1 (Just A) = ((), Q2, Pop)
    transition B Q2 (Just A) = ((), Q2, Pop)
    transition B Q2 Nothing = ((), QAccept, NullOp)
    transition _ _ _ = ((), Q0, NullOp)


--------------------------------------------------
-- * Turing Machine


data TMOp s =
  EnqL s
  | DeqR
  | TMNullOp
  deriving (Eq, Show)

-- | Turing Machine with Queue
newtype TM a b state symbol = TM {unTM :: MSF (State (state, Seq symbol)) a b}

-- | Construct a TM from a transition function
tm :: (a -> state -> Maybe symbol -> (b, state, TMOp symbol)) -> TM a b state symbol
tm f = let m = MSF $ (\a -> do
                        (s, q) <- get
                        let (b, s', op) = f a s (viewl q)
                        case op of
                          EnqL new -> put (s', (Seq.<| q) new)
                          DeqR -> put (s', dropr 1 q)
                          TMNullOp -> pure ()
                        pure (b, m))
  in TM m

runTM :: TM a b state symbol -> [a] -> state -> Seq symbol -> ([b], (state, Seq symbol))
runTM (TM msf) input initialState initialQueue = runState (go input msf) (initialState, initialQueue)
  where
    go [] _ = pure []
    go (x:xs) m = do
      (b, m') <- unMSF m x
      bs <- go xs m'
      pure (b:bs)


anbncn :: TM Sym () Q Sym
anbncn = tm transition
  where
    transition :: Sym -> Q -> Maybe Sym -> ((), Q, TMOp Sym)
    transition A Q0 _ = ((), Q0, EnqL A)
    transition B Q0 (Just A) = ((), Q1, EnqL A)


--------------------------------------------------
-- Go

main :: IO ()
main = do
  -- FSM
  putStrLn "----------"
  putStrLn "-- FSM"
  let str = "aaabaa"
  print $ runFSM countA str 0


  -- PDA
  putStrLn "----------"
  putStrLn "-- PDA"
  let strings = [ [A, A, B, B],
                  [A, A, B, B, B],
                  [A, A, A, B, B]
                ]
  mapM_ (\str -> do
    let (out, (state, stack)) = runPDA anbn str Q0 Seq.empty
    putStrLn $ show str
      ++ " : " ++ show out
      ++ " : " ++ show state
      ++ " : " ++ show stack) strings


  -- TM
  putStrLn "----------"
  putStrLn "-- Turing Machine"
  let strings = [
        [A],
        [A, B],
        [A, B, C],
        [A, B, C, C],
        [A, B, B, C],
        [A, A, B, B, C, C],
        [A, A, B, B, B, C, C],
        [A, A, A, B, B, C, C, C]
        ]
  mapM_ (\str -> do
    let (out, (state, queue)) = runTM anbncn str Q0 Seq.empty
    putStrLn $ show str
      ++ " : " ++ show out
      ++ " : " ++ show state
      ++ " : " ++ show queue) strings
