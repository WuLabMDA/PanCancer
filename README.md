# PanCancer

The codes are designed to extract shape descriptor and regional variation features. 

If you are using the codes, please cite the following paper and code DOI:
1. Wu et al. Radiological tumor classification across imaging modality and histology, Nature Machine Intelligence, 2021.
2. https://doi.org/10.5281/zenodo.4906510

"matlab" folder contains codes for shape features and "python" folder contains codes for regional variation features

matlab folder contains the following items:
1. main.m:  this is the main function to extract shape features, please follow the instructions to extract shape features
2. Function: this folder contains two used defined functions
   --checkBIM.m: checks the quality of surfact point cloud
   --spharm_coeff.m: extract 2nd order shaper features from spherical harmonic decomposition
3. Process_data: this folder saves the input data, we provide an example "Decathlon"
4. SPHARM_output: this folder saves the output data, where each subfolder corresponds to one step
5. third_party_package: this folder contains the SPHARM function provided by Dr. Li Shen's group (https://www.med.upenn.edu/shenlab/spharm-mat.html)
6. SPHARM-MAT.pdf: this file contains details of SPHARM function


python folder contains the following items:
1. main.py: this is the main function to extract regional variation features, please follow the instructions inside.
2. utils.py: this contains user defined functions called in main.py
3. Data: this folder provides the input data, we provide an example "TCGA"

If you have any questions/suggestions, please contact Jia Wu, Assistant Professor, MD Anderson Cancer Center, jwu11@mdanderson.org
