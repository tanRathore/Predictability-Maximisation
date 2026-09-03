<code> pip install -r requirements.txt </code>

# Predictiability-maximization-Tanishq

Test whether *predictiability maximization* and DLM together can explain the distribution of structural preferences in SOV as well as SVO languages.

Methods:
- Generate random baseline trees that match natural language trees in terms of predictiability and DL distribution
- Do we get Hindi-like word orders when both predictability and DL are controlled?
- Inspired by methods in Yadav et al Open Mind paper (Intervener complexity)

Data:
- Surface-syntactic universal dependencies (SUD)


Tasks:
- Define predictability measures for a dependency: Number of interverners that directly modify the preceding co-depenent. For example, in Mohan-ne kabir-ko kitaab di. The dependency between Mohan and di has two interveners that directly modify "di" (preceding co-dependent).
- Define a function for predictability measure in Measures class file:
- An algoritghm that generates trees that minimize for DL with degree $d$ and maximize for predictability for degree $p$  
- Write random baseline generation code for predctability-matched trees: See DL-matched baseline code for reference
- Write random baseline code 

## Word Order Distribution Comparison

This repository now includes an experiment to compare word order distributions across multiple languages (e.g., Chinese, English, Czech, Arabic, Hindi). The experiment performs the following tasks:

1. **Extraction of Word Orders:**
   - Extracts word order sequences from SUD files using the `extract_orders_from_sud` function.
   - Each sentence is represented as a sequence of the first letters of the 'upostag' field (e.g., "NNNV", "NVNN").

2. **Probability Distribution & Entropy:**
   - Computes the probability distribution of word orders for each language.
   - Calculates entropy for each language's word order distribution.

3. **KL Divergence & Euclidean Distance:**
   - Computes pairwise KL divergence (cross-entropy) between the distributions of different languages.
   - Computes pairwise Euclidean distances between representative word orders.
   - This allows us to compare, for example, how far Czech is from Hindi, and verify if Arabic is closer to English than Hindi.

4. **Visualization:**
   - Displays heatmaps for pairwise KL divergence and Euclidean distances.
   - Produces bar charts for each language’s word order distribution.

