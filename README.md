# TreeCutterPy
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20395899.svg)](https://doi.org/10.5281/zenodo.20395899)

This is a python package designed to help you cut and reclassify TLS and ALS points.
It does works for already segmented trees using an automatic algorithm, and is an tool for refinement.

## How to use it:

1. Install the package,

2. The code is made to work with already segmented trees with an authomatic algorithm/model, and each tree has an ID, for example PredInstance of single trees from SegmentAnyTree model, so if you have any other name for your single tree id, just change it when the program asks for it. 

The code is generalized and it will ask for directories at the start. The input folder need to contain .laz files with multiple trees inside (for example a plot), (is fine if there are multiple .laz files) and it is probably useless if you have a single tree for each file, since in that case you cannot see neighbour trees to refine the segmentation.

3. Install dependencies: `pip install numpy==2.2.5 laspy==2.5.4 matplotlib==3.10.1 PyQt5==5.15.11` 

4. (Optional but recommended) Create a virtual environment to avoid dependency conflicts: `python -m venv .venv && .venv\Scripts\activate`. Dependencies are installed automatically by step 4.

5. For citation

## How to cite

If you use this software in academic work, please cite:

> Papucci, E., Yrttimaa, T. (2026). *TreeCutterPy: interactive point-cloud tree cutter and reclassifier* (Version 1.6.0). Zenodo. https://doi.org/10.5281/zenodo.20395899
