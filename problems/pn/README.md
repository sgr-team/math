# Permutation Neuron

## Overview

The Permutation Neuron is a computational unit that implements a permutation-based transformation of input signals. The neuron maintains a set of internal vectors that are reordered based on their interaction with the input data. This reordering process maps the input space to a discrete set of output patterns, where each pattern corresponds to a specific permutation of the internal vectors.

## Formulation

Given an input vector X and internal vectors (V0, V1, ...), the neuron computes:

1. Vector products: \(p_0 = X · V0, p_1 = X · V1, ...)
2. Sorting: p_0, p_1, ...
3. Permutation mapping: The neuron maintains a set of output vectors, one for each possible permutation of the internal vectors. When a specific permutation is detected (based on the sorted order of vector products), the corresponding output vector is activated. This mapping is learned during training, where each permutation is associated with a specific output pattern that best represents the desired classification or transformation.

## Setup

- Create a .data folder in this directory.
- Place two CSV files in the .data folder: train.csv and test.csv.
    - The CSV files can include headers or be headerless.
    - The first column in each row must represent the label.

## Scripts

### GA

The script launches a genetic algorithm.

```bash
mkdir .output && cargo run --example ga >> .output/.log
```
