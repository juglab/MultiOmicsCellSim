# Multi Omics Cell Simulator

This package contains a simulator that is able to generate multi-modal data (i.e., microscopy-like, spatial-omics-like, single-cell-sequencing-like) from a set of input parameters.
Please keep in mind that the aim of this simulator is **not** to attain an higher degree of biological fidelity. Instead, the main advantage is to have a controllable tool that allows to mimic some underlying processes that govern biological data in a principled way, i.e., knowing the hidden "ground-truth" that is governing the correlations between the different modalities.


## Installation

1) Create your conda or venv environment:
```
python -m venv venv && source venv/bin/activate
```
2) To install the package locally from GitHub, clone the repository and install the package in development mode:

```
pip install -e git+https://github.com/juglab/MultiOmicsCellSim.git
```
If you get an error regarding a missing seuptool-based build, please upgrade your pip installation.

### Dev-Mode

Just pull the repository, cd into it and install with `python -m pip install -e .`. Again, ensure you have an updated version of pip which supports `hatch` and editable mode together.

## Features:

- Tissue-level pictograms (WIP): Generate **microscopy-like** images starting from some input features at a tissue, cell and subcellular level. 
