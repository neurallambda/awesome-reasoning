{-

USAGE:
cabal run machine -- -i rules/anbn.json -g 20
# or
cabal build
dist/build/machine/machine --input anbn.json --generations 20
-}

{-# LANGUAGE RecordWildCards #-}

module Main where

import Options.Applicative

import qualified Data.Sequence as Seq
import qualified Data.Map as M
import qualified Data.ByteString.Char8 as B8

import qualified Machine as Mch

data CLIOptions = CLIOptions
  { inputFile :: FilePath
  , numGenerations :: Int
  }

cliOptions :: Parser CLIOptions
cliOptions = CLIOptions
  <$> strOption
    ( long "input"
    <> short 'i'
    <> metavar "FILE"
    <> help "Input file containing transition table"
    )
  <*> option auto
    ( long "generations"
    <> short 'g'
    <> metavar "INT"
    <> value 20
    <> showDefault
    <> help "Number of generations to produce"
    )

main :: IO ()
main = do
  options@CLIOptions{..} <- execParser $ info (cliOptions <**> helper)
    ( fullDesc
    <> progDesc "Generate sentences from a given grammar"
    <> header "grammar-generator - a sentence generator for formal grammars"
    )

  -- Read transition table from input file
  jsonInput <- B8.readFile inputFile
  case Mch.parseMachineSpec jsonInput of
    Left err -> putStrLn $ "Error parsing machine specification: " ++ err
    Right machineSpec -> do
      let Mch.MachineSpec{..} = machineSpec
          -- trans = M.fromList rules
          initialState = Mch.PDAS Mch.Q0 Seq.empty
          strings = Mch.pdaString (Mch.untransition rules) Mch.halt symbols initialState

      putStrLn $ "Machine: " ++ show machine
      putStrLn "generations:"
      mapM_ print (take numGenerations strings)
      putStrLn "done."
