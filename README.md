# TreeCutterPy

This is a small code designed to help you cut and reclassify points.
It does works for already segmented trees using an automatic algorithm, and is an tool for refinement. It is still slow, working on optimization and a new version in C++.

## How to use it:

1. Clone the repository,

2. The code is made to work with already segmented trees with an authomatic algorithm/model, and each tree has an ID, for example PredInstance of single trees from SegmentAnyTree model, so if you have any other name for your single tree id, just change it when the program asks for it. The code is generalized at the moment will ask for directories at the start. The input folder can contain .laz files with multiple trees inside (for example a plot), (is fine if there are multiple .laz files) and it is probably useless if you have a single tree for each file. 

3. To start the code, open it with any system, e.g. Visual Studio Code, Spider, etc...

4. Install dependencies: `pip install numpy==1.26.4 laspy==2.5.1 matplotlib==3.8.2 PyQt5==5.15.10` (maybe is good if you make a virtual environment for the packages, so you don't make conflicts up) 

5. Enjoy :D

A kind thanks for Tuomas and Carl (Co-Authors)