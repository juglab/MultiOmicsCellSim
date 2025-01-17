# Multi Omics Cell Simulator

This package contains a simulator that is able to generate multi-modal data (i.e., microscopy-like, spatial-omics-like, single-cell-sequencing-like) from a set of input parameters.
Please notice that the aim of this simulator is **not** to attain an higher degree of biological fidelity. 
Instead, the main advantage is to have a controllable tool that allows to mimic some underlying processes that govern biological data in a principled way, i.e., knowing the hidden "ground-truth" that is governing the correlations between the different modalities.


## Installation

1) Create your conda or venv environment:
```
python -m venv venv && source venv/bin/activate
```
2) To install the package in development mode, update your pip and run:

```
git clone https://github.com/juglab/MultiOmicsCellSim.git
cd MultiOmicsCellSim
pip install -e .
```
## Quick Start

Refer to the [example notebook](examples/example.ipynb)

## Features:

- Stochastical initial state generation: A Guideline (a curve definining a spatial distribution when a cell could spawn) is randomly generated, and it allows to sample cells with a particular distribution of cell types
- Growing Cells: Cell Pott Model-based growing of cells.
- Subcellular Representation: Subcellular content is represented using a Reaction Diffusion model. This allows to represent biological diversity in cell phenotypes that can be further encoded as genotypes.

