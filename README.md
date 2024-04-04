# Neurallambda Data: On Reasoning

**Purpose:** The [`neurallambda`](https://github.com/neurallambda/neurallambda) project needs to prove it is superior to traditional ML architectures, and so we need datasets for training and benchmarking. This repo tracks that data oriented effort of the `neurallamba` project.

* **TODO:**
  * [ ] Build toy data for variable-abstraction multi-hop problems
  * [ ] Generate Wikidata Multi-hop Dataset
  * [ ] Build NLP version of Raven's Progressive Matrices?
  * [ ] Build NLP version of Wisconsin Card Sorting?


## Data Landscape

An attempt to understand the datasets that have come before us.


| Name                            | Description                                | Link                                                                            |
|---------------------------------|--------------------------------------------|---------------------------------------------------------------------------------|
| TOY PROBLEMS                    |                                            |                                                                                 |
| word ladder                     |                                            |                                                                                 |
| parser                          |                                            |                                                                                 |
| longest cmn subseq              |                                            |                                                                                 |
| string reversal                 |                                            |                                                                                 |
| wisconsin card sorting          |                                            |                                                                                 |
| anagram                         |                                            |                                                                                 |
| palindrome                      |                                            |                                                                                 |
| (*) Big Bench Hard              | 23 challenges (only 6k datapoints)         | https://github.com/suzgunmirac/BIG-Bench-Hard                                   |
| logical entailment dataset      | logic strings by deepmind                  | https://huggingface.co/datasets/tasksource/logical-entailment                   |
| logical entailment dataset code | (generate it yourself)                     | https://github.com/google-deepmind/logical-entailment-dataset                   |
| AB-XY Game                      |                                            |                                                                                 |
| FSM Game                        | generate strings according to grammar      |                                                                                 |
| Adaptive Grammar                | grammar rule might change                  |                                                                                 |
|                                 |                                            |                                                                                 |
| AGENT/TOOL                      |                                            |                                                                                 |
| THUDM AgentInstruct             | long form dialogs                          | https://huggingface.co/datasets/THUDM/AgentInstruct                             |
| WANG AgentInstruct              |                                            | https://huggingface.co/datasets/WangResearchLab/AgentInstruct                   |
| KnowLM Tool                     | prompt + tool call + answer                | https://huggingface.co/datasets/zjunlp/KnowLM-Tool                              |
| Glaive Tool Usage               | sys prompt says tools + prompt + answer    | https://huggingface.co/datasets/roborovski/glaive-tool-usage-dpo                |
| opentoolformer retrieval        | prompt + tool call                         | https://huggingface.co/datasets/kenhktsui/open-toolformer-retrieval             |
|                                 |                                            |                                                                                 |
| CODE                            |                                            |                                                                                 |
| rosetta                         |                                            | https://huggingface.co/datasets/cakiki/rosetta-code                             |
| EvoEval Tool Use                | 100 prompt + code + tests                  | https://huggingface.co/datasets/evoeval/EvoEval_tool_use                        |
|                                 |                                            |                                                                                 |
| MATH/LOGIC                      |                                            |                                                                                 |
| MetaMath                        | one-shot math                              | https://github.com/meta-math/MetaMath                                           |
| MetaMathFewShot                 | few-shot math                              | https://huggingface.co/datasets/abacusai/MetaMathFewshot                        |
| MathPile                        | 9B tok from filtered internet              | https://huggingface.co/datasets/GAIR/MathPile                                   |
| (*) LogiQA                      | NL multi choice, requires abstraction      | https://github.com/lgw863/LogiQA-dataset                                        |
|                                 |                                            |                                                                                 |
|                                 |                                            |                                                                                 |
| AGI/CAUSALITY                   |                                            |                                                                                 |
| (*) im a strange dataset        | Tough for LLMs because of self-references. | https://github.com/TristanThrush/i-am-a-strange-dataset                         |
| DiagGSM8k                       | NL Reasoning Benchmark                     | https://github.com/dvlab-research/MR-GSM8K                                      |
| CLadder                         | Causal reasoning                           | https://huggingface.co/datasets/causalnlp/CLadder                               |
| Cause-Effect Pairs              | 108 datasets of 2 var dyanmics (not NL)    | https://webdav.tuebingen.mpg.de/cause-effect/                                   |
|                                 |                                            |                                                                                 |
|                                 |                                            |                                                                                 |
| NATURAL LANGUAGE                |                                            |                                                                                 |
| UltraInteract_sft               | GPT generated iterated reasoning dialogs   | https://huggingface.co/datasets/openbmb/UltraInteract_sft                       |
| MUD videogames                  | (various could be training data)           |                                                                                 |
| Winogrande                      | ambiguous sentences, fill in 1 word        | https://huggingface.co/datasets/winogrande                                      |
| Winograd_wsc                    | ambiguous sentences, choose the right word | https://huggingface.co/datasets/winograd_wsc                                    |
| Contradiction                   | 2 phrases, do they contradict              | https://www-nlp.stanford.edu/projects/contradiction/                            |
| Recognizing Textual Entailment  | 2 phrases, do they entail each other       | https://github.com/hltfbk/EOP-1.2.1/wiki/Data-Sets                              |
| Textual Entailment Pool         | more entailment                            | https://www.aclweb.org/aclwiki/index.php?title=Textual_Entailment_Resource_Pool |
| Answer Validation               | 2 phrases, does the answer solve question  | http://nlp.uned.es/clef-qa/repository/ave.php                                   |
| Monotonicity Entailment         | x is true, does y follow                   | https://huggingface.co/datasets/tasksource/monotonicity-entailment              |
| entailment                      | passage, question -> T/F                   | https://huggingface.co/datasets/nc33/entailment                                 |
| Commonsense QA                  |                                            |                                                                                 |
| GLUE                            | several datasets                           | https://huggingface.co/datasets/nyu-mll/glue                                    |
| custom multi-hop                | use wikipedia's graph of articles          |                                                                                 |
|                                 |                                            |                                                                                 |
| PARSING                         |                                            |                                                                                 |
| MNLI Entailment                 | sentence parsing + entailment              | https://huggingface.co/datasets/westphal-jan/mnli_entailment                    |
|                                 |                                            |                                                                                 |
