# TreeCutterPy

This is a small code designed to help you cut and reclassify points.
It does works for already segmented trees using an automatic algorithm, and is an tool for refinement. It is still slow, working on optimization and a new version in C++.

## How to use it:

1. Clone the repository,

2. The code is made to work with trees segmented with SegmentAnyTree model (Ref.), and it use PredInstance of single trees to work, so if you have any other name for your single tree id, just change that within the code. The code is not generalized at the moment or an up, is just a code, so to make it work please change the directory at the bottom for input and output. The input folder need to contain .laz file with multiple trees inside (for example a plot), so it doesn't work if you have saved single trees. 

3. Install dependencies (if any): `pip install numpy==1.26.4 laspy==2.5.1 matplotlib==3.8.2 PyQt5==5.15.10` (maybe good if you make a virtual environment for the packages, so you don't make conflicts up"

4. run the script from your gui, I use Visual Studio Code. 
