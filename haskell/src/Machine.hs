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
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE LambdaCase #-}

module Machine where

import Data.Sequence ( Seq, ViewL(..), (<|), Seq(..), (|>) )
import qualified Data.Sequence as Seq
import Data.Foldable ( toList, foldl' )
import Data.List (intercalate)
import qualified Data.Map as M
import Data.Kind (Type)
import Data.Map.Extra (lookupMatchAny, Any(..), MatchAny(..))
import Data.Maybe (mapMaybe)
import Data.Aeson
import Data.ByteString (ByteString)
import Data.Aeson.Types (Parser)
import Data.Text (unpack)
import qualified Data.Vector as Vector
import qualified Data.Text as T
import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as B8
import Data.Either (fromRight)

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
          haltingAccs = [toList acc | halt state]
        in
          haltingAccs ++ bfs newQueue

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
  -> Maybe (R m a s, S m a s) -- (transition value, new state)
runStep table st input =
  case lookupMatchAny (mkL input st) table of
    Just transition -> Just (transition, action transition st)
    Nothing -> Nothing -- no transition found

-- | Run a machine on a list of input symbols
runMachine :: (Machine m a s
              , Ord (L m a s)
              , Show (L m a s)
              , MatchAny (L m a s)
              )
  => M.Map (L m a s) (R m a s) -- transition table
  -> S m a s -- initial state
  -> [a] -- input symbols
  -> Maybe (R m a s, S m a s)
runMachine table initialState = foldl' f $ Just (error "empty input", initialState)
  where
    f (Just (_, state)) = runStep table state
    f Nothing = const Nothing


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
-- * JSON

newtype Transition = Transition (L PDA Char (Q, Char), R PDA Char (Q, Char))
  deriving (Show, Eq)

instance FromJSON Transition where
  parseJSON = withArray "Transition" $ \arr -> do
    input <- arr `parseAt` 0
    fromState <- arr `parseAt` 1
    stackTop <- arr `parseAt` 2
    toState <- arr `parseAt` 3
    action <- arr `parseAt` 4

    l <- PDAL <$> parseInput input <*> pure (A fromState) <*> parseMaybeAny stackTop
    r <- PDAR toState <$> parseAction action

    return (Transition (l, r))

    where
      parseAt arr i = parseJSON (arr Vector.! i)

      parseInput :: Value -> Parser (Any Char)
      parseInput (String "^") = return (A '^')
      parseInput (String "$") = return (A '$')
      parseInput (String [c]) = return (A c)
      parseInput other = fail $ "Invalid input: " ++ show other

      parseMaybeAny :: Value -> Parser (Maybe (Any Char))
      parseMaybeAny Null = return Nothing
      parseMaybeAny v = do
        c <- parseChar v
        return (Just c)

      parseChar :: Value -> Parser (Any Char)
      parseChar = withText "Char" $ \t ->
        case T.unpack t of
          [c] -> return (A c)
          other -> fail $ "Invalid character: " ++ other

parseAction :: Value -> Parser (PDAOp Char)
parseAction (String "nullop") = return NullOp
parseAction (String "pop") = return Pop
parseAction v = pushParser v
  where
    pushParser = withArray "Push" $ \arr -> do
      op <- arr `parseAt` 0
      case op of
        String "push" -> do
          symbol <- arr `parseAt` 1
          case unpack symbol of
            [c] -> return $ Push c
            other -> fail $ "Invalid push symbol: " ++ other
        _ -> fail "Invalid push action"

    parseAt arr i = parseJSON (arr Vector.! i)

parseTransitions :: ByteString -> Either String [Transition]
parseTransitions = eitherDecodeStrict'

data MachineType = PDA | TM | DFA
  deriving (Show, Eq)

instance FromJSON MachineType where
  parseJSON = withText "MachineType" $ \t ->
    case t of
      "pda" -> return PDA
      "tm" -> return TM
      "dfa" -> return DFA
      _ -> fail "Invalid machine value"

data MachineSpec = MachineSpec
  { machine :: !MachineType
  , symbols :: ![Char]
  , rules :: ![Transition]
  }
  deriving (Show, Eq)

instance FromJSON MachineSpec where
  parseJSON = withObject "MachineSpec" $ \obj -> do
    machine <- obj .: "machine"
    rules <- obj .: "rules"
    symbols <- obj .: "symbols" >>= parseSymbols
    transitions <- mapM parseJSON rules
    return (MachineSpec machine symbols transitions)
    where
      parseSymbols = withArray "symbols" $ \arr ->
        mapM parseSymbol (toList arr)

      parseSymbol = withText "Symbol" $ \t ->
        case T.unpack t of
          [c] -> return c
          _ -> fail "Invalid symbol"

-- instance FromJSON MachineSpec where
--   parseJSON = withObject "MachineSpec" $ \obj -> do
--     machine <- obj .: "machine"
--     rules <- obj .: "rules"
--     symbols <- obj .: "symbols"
--     transitions <- mapM parseJSON rules
--     return (MachineSpec machine symbols transitions)

parseMachineSpec :: ByteString -> Either String MachineSpec
parseMachineSpec = eitherDecodeStrict'

-- exStr =
-- [
--   ["^", "Q0", null, "Q1", ["push", "$"]],
--   ["$", "Q1", "$", "QAccept", "pop"],
--   ["a", "Q1", "$", "Q1", ["push", "A"]],
--   ["a", "Q1", "A", "Q1", ["push", "A"]],
--   ["A", "Q1", "A", "Q1", ["push", "A"]],
--   ["x", "Q1", "A", "Q2", "nullop"],
--   ["b", "Q1", "A", "Q2", "pop"],
--   ["b", "Q2", "A", "Q2", "pop"],
--   ["$", "Q2", "$", "QAccept", "pop"]
-- ]


transitionTable :: String
transitionTable = unlines
  [ "["
  , "  [\"^\", \"Q0\", null, \"Q1\", [\"push\", \"$\"]],"
  , "  [\"$\", \"Q1\", \"$\", \"QAccept\", \"pop\"],"
  , "  [\"a\", \"Q1\", \"$\", \"Q1\", [\"push\", \"A\"]],"
  , "  [\"a\", \"Q1\", \"A\", \"Q1\", [\"push\", \"A\"]],"
  , "  [\"A\", \"Q1\", \"A\", \"Q1\", [\"push\", \"A\"]],"
  , "  [\"x\", \"Q1\", \"A\", \"Q2\", \"nullop\"],"
  , "  [\"b\", \"Q1\", \"A\", \"Q2\", \"pop\"],"
  , "  [\"b\", \"Q2\", \"A\", \"Q2\", \"pop\"],"
  , "  [\"$\", \"Q2\", \"$\", \"QAccept\", \"pop\"]"
  , "]"
  ]

untransition :: Functor f => f Transition -> f (L PDA Char (Q, Char), R PDA Char (Q, Char))
untransition xs = f <$> xs
  where f (Transition x) = x

-- anbnTransitions = unTransition <$> right
--   where
--     unTransition (Transition x) = x
--     Right right =  parseTransitions $ B8.pack transitionTable


instance FromJSON Q where
  parseJSON = withText "Q" $ \t ->
    case t of
      "Q0" -> return Q0
      "Q1" -> return Q1
      "Q2" -> return Q2
      "Q3" -> return Q3
      "Q4" -> return Q4
      "QAccept" -> return QAccept
      "QReject" -> return QReject
      _ -> fail "Invalid Q value"


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
  deriving (Eq, Show, Ord, Read)

instance MatchAny Q where matchAny x y = x == y


----------
-- * Go

showSeq :: Show a => Data.Sequence.Seq a -> String
showSeq xs = "{" ++ intercalate ", " (show <$> toList xs) ++ "}"

formatPDAResult :: Maybe (R PDA Char (Q, Char), S PDA Char (Q, Char)) -> String
formatPDAResult Nothing = "no transition found, and didn't reach halting state"
formatPDAResult (Just (_, PDAS finalState stack)) =
  "Final state: " ++ show finalState ++ " | Stack: " ++ showSeq stack

halt :: S PDA a (Q, stack) -> Bool
halt (PDAS QReject _) = True
halt (PDAS QAccept _) = True
halt _ = False
