# MRI-Based DAT Uptake Assessment in Parkinson's Disease
Dopamine transporter (DAT) imaging is commonly used for monitoring Parkinson’s disease (PD), where DAT uptake amount is computed to assess PD severity. However, DAT imaging has a high cost and the risk of radiance exposure and is not available in general clinics. Recently, MRI patch of nigral region has been proposed
as a safer and easier alternative. 
This repository offers a symmetric regressor for predicting the DAT uptake amount from nigral MRI patch.
Note that, the susceptibility-map-weighted-imaging (SMWI) technique of MRI is used.
DAT uptake amount is commonly expressed as the specific binding ratio (SBR).

## Input Processing ##
From SMWI of the brain, extract right and left nigral patches where each patch encompasses a volume of `50x50x20` voxels centered at the corresponding nigrosome-1's centroid.
Pass these patches to the predictor to obtain the SBR score.
(Note that, the SMWI patch intensity will be normalized internally in our code by dividing it by the mean intensity per patch. Therefore, external normalization is not required.)

## Usage ##
Load the MAT file containing the nigral patches\
`d = io.loadmat('nigral_patch_example.mat')`

Create the desired regressor model (defaule: symmetric) and predict SBR\
`model = sd.SMWI_DAT(model=sd.constants.model_type_symmetric)`\
`right_sbr, left_sbr = model.predict_sbr(d['right_patch'], d['left_patch'])`\
`print(right_sbr[0][0], left_sbr[0][0])`

## Dependency ##
`tensorflow 2.9.0`
`matplotlib 3.4.2`
