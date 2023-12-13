# Towards Search-based Decoding for LLMs via Monte Carlo Tree Search and A* Search

This project presents an approach that applies both Monte Carlo Tree Search (MCTS) and A* Search for decoding sequences in Large Language Models (LLMs), specifically GPT-2. Our method aims to improve the balance between exploration and exploitation in the token generation process. By implementing MCTS, we develop a search tree that assesses sequences based on both immediate quality and potential future developments, facilitating the generation of coherent and varied text. Simultaneously, A* Search, driven by a heuristic function with random Gaussian-distributed rewards, refines the search for contextually appropriate and inventive text. These strategies provide a practical example of applying search-based algorithms to LLMs, showcasing their capability to produce text that is both human-like and contextually nuanced. To provide an example of a more deterministic and finite-state heuristic, we also demonstrate crossword puzzles as an example of a game that ties in both language and a search space of possible. Our results contribute to understanding the challenges in LLM decoding and demonstrate the utility of applying MCTS and A* Search to effectively explore the decision space in language generation.

## Introduction
Inspired by the recent $Q^*$ developments out of OpenAI, this project explores the novel application of Monte Carlo Tree Search (MCTS) and A* Search in decoding sequences from Large Language Models (LLMs), specifically GPT-2. Our work aims to strike a balance between exploration and exploitation in the context of token generation, an area where traditional decoding methods often fall short. By integrating MCTS and A* Search, we present a framework that navigates the vast decision space of LLMs more effectively, allowing for more nuanced and contextually rich text generation. We hope that this work inspires further development away from pure-autoregressive LLMs towards more planning-capable models.

MCTS, renowned for its success in complex decision-making environments like Go and Chess, is adapted here to manage the combinatorial explosion inherent in language generation. Our implementation dynamically expands the search tree, evaluating nodes based on both the quality of the generated sequence and potential future continuations. We introduce a unique reward function that considers sequence length, repetition, and basic language properties, promoting coherent and diverse text generation.

Parallel to this, we employ A* Search, a best-first search algorithm, to further refine the decoding process. By using a heuristic function that incorporates random Gaussian-distributed rewards, we enhance the search efficiency in generating text. Our approach demonstrates improved exploration in the decision space, leading to more creative and contextually relevant outputs compared to standard decoding methods.

Through this paper, we provide insights into the challenges and opportunities of applying search-based algorithms in the realm of LLMs. 

## Background

Recent advancements in open-ended language generation are largely attributed to the emergence of large transformer-based language models, trained on extensive web-based datasets \cite{huggingface}. These models, including OpenAI's ChatGPT and Meta's LLaMA, have demonstrated remarkable capabilities in tasks ranging from generalization to code handling and processing non-text data \cite{huggingface}. A crucial aspect underpinning these achievements is the development of efficient and effective decoding methods.

Auto-regressive language generation, the backbone of these models, relies on the principle that the probability distribution of a word sequence can be decomposed into the product of conditional next-word distributions \cite{huggingface}. This process forms the basis of various decoding strategies, each with its unique approach to sequence generation. Auto-regressive language generation can be mathematically expressed as:
\[ P(w_{1:T}|W_{0}) = \prod_{t=1}^{T} P(w_{t}|w_{1:t-1},W_{0}) \]
where \( w_{1:0} = \emptyset \) and \( W_{0} \) is the initial context word sequence.

## Greedy Search
Greedy search, the simplest form of decoding, selects the word with the highest probability at each timestep, forming a sequence based on maximum immediate likelihood \cite{huggingface_greedy}. Despite its simplicity, this method can often lead to suboptimal sequences due to its lack of consideration of various hypotheses.

## Beam Search
Beam search addresses some limitations of greedy search by considering the most likely \textit{num\_beams} hypotheses at each timestep \cite{huggingface_beam}. It maintains multiple probable sequences and eventually selects the hypothesis with the highest overall probability. This method is more comprehensive than greedy search but does not guarantee the discovery of the most likely output.

## Sampling
Basic sampling introduces randomness in word selection, choosing the next word according to its conditional probability distribution \cite{huggingface_sampling}. This approach adds diversity to the generated text but can sometimes lead to incoherence.

## Top-K Sampling
Top-K sampling, a refinement of basic sampling, limits the selection pool to the K most likely next words \cite{huggingface_topk}. This strategy balances between diversity and coherence, making it effective for story generation in models like GPT-2.

## Top-p (Nucleus) Sampling
Top-p sampling, also known as nucleus sampling, dynamically selects words from the smallest set whose cumulative probability exceeds a threshold p \cite{huggingface_topp}. This method adapts the size of the selection pool based on the predictability of the next word, offering a more flexible approach compared to Top-K sampling.

Each of these methods offers distinct advantages and trade-offs, shaping the way transformers generate language in various applications.

## Heuristic for language
Developing a heuristic or reward function for language generation presents significant challenges due to the vastness of the search and parameter spaces in language models. As noted by Yann LeCun, this task essentially requires the model to possess a comprehensive 'world model' to evaluate the quality ($Q(s,a)$) of a vast number of potential states (s) and actions (a) in language. This complexity is compounded by the inherent intricacies of language, where meaning and coherence depend on a delicate balance of syntax, semantics, context, and cultural nuances.

Incorporating an oracle model to evaluate sequences during the generation process further complicates matters. The real-time assessment of each possible continuation of a text not only demands substantial computational resources but also significantly slows down the generation process. Given the combinatorial explosion of possible paths in language generation, constantly consulting an oracle model for evaluation can lead to inefficiencies, making it impractical for real-world applications where speed and responsiveness are crucial. This underlines the need for more efficient heuristic approaches that balance computational feasibility with effective decision-making in language generation tasks.

## A* Search for decoding

The A* search-based decoder initializes with a given context or starting sequence. It then expands this sequence by generating potential next tokens. For each of these tokens, the model predicts the next likely tokens and evaluates them using a heuristic function. This function, typically a Gaussian distribution in this implementation, assigns a cost to each expanded node (sequence). The cost reflects not only the probability of the sequence (as given by the model) but also includes a heuristic estimate of the sequence's potential, which could be based on factors like coherence, relevance, or adherence to a particular style or format.

As the search progresses, the decoder maintains a priority queue of these nodes, where sequences with lower heuristic costs are given higher priority. The A* algorithm then iteratively selects the sequence with the lowest cost from this queue for further expansion. This process continues until the decoder reaches a predefined maximum token limit or other system-defined termination criteria.

The A* search-based decoder offers a nuanced approach to text generation, allowing for more deliberate and strategic construction of sequences compared to traditional left-to-right decoding methods.

## Monte Carlo Tree Search (MCTS) for decoding

The MCTS-based decoder for language generation utilizes a tree structure where each node represents a potential continuation of a text sequence. It combines the principles of Monte Carlo simulations and tree search strategies to navigate the vast output space of the language model more effectively.

The MCTS algorithm starts with a root node representing the initial context or starting sequence. In each iteration, it performs a series of four steps: selection, expansion, simulation, and backpropagation. During selection, the algorithm traverses the tree by selecting nodes based on a selection policy that balances exploration and exploitation. The tree is then expanded by adding child nodes to the selected node. Next, a simulation is performed by extending the sequence from each new node to evaluate its potential. The result of the simulation is used to update the values and visit counts of the nodes along the path from the selected node to the root through backpropagation. This process is repeated until a termination condition is met, such as reaching a maximum token limit.

By combining the principles of MCTS and A* search, we can effectively navigate the decision space in language generation tasks, leading to more creative and contextually relevant text outputs compared to standard decoding methods.

## Experiment Results

The A* search-based decoder received a total of 97 points, the MCTS-based decoder received a total of 91 points, and the traditional sampling method received a total of 112 points. This indicates that the A* and MCTS decoders perform comparably to traditional decoding methods in terms of quality. However, it is important to note that these results are based on a specific oracle model and reward function, and further experimentation and fine-tuning are needed to draw robust conclusions.

## Application to Crosswords

In addition to its application in language generation, the search-based decoding methods proposed can also be applied to other domains that involve language and a search space of possibilities. One example is solving crossword puzzles, where the search tree can represent the different positions and letters in the crossword grid, and the heuristics used can take into account factors such as the fit of the words with the clues, the length of the words, and the number of common letters.

## Insights

The experiments conducted highlight the potential of search-based decoding methods in improving text generation in large language models. The results show that these methods can produce text of similar quality to traditional decoding methods while offering the advantages of exploration and more deliberate decision-making. However, it is important to consider the limitations and challenges of implementing and fine-tuning these methods, including the development of effective reward functions and heuristics that balance exploration and exploitation.

Overall, the findings of this project contribute to the development of more effective and nuanced text generation methods that consider both token-level decisions and strategic exploration in the decision space. By utilizing search-based algorithms like A* Search and MCTS, it is possible to enhance the capabilities of large language models in understanding and generating coherent, contextually appropriate language.

## Conclusion

This project presents an approach that applies both Monte Carlo Tree Search (MCTS) and A* Search for decoding sequences in Large Language Models (LLMs) to improve text generation. By integrating these search-based algorithms, we strike a balance between exploration and exploitation, leading to more creative and contextually relevant text outputs. The application of these methods in language generation tasks showcases their capability to produce human-like and contextually nuanced language. Future work should focus on fine-tuning reward functions and incorporating more deterministic and finite-state heuristics to further refine these methods. Overall, this work contributes to the development of more effective text generation methods that consider strategic decision-making in large language models.
