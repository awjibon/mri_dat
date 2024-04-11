# MRI-Based DAT Uptake Assessment in Parkinson's Disease
Dopamine transporter (DAT) imaging is commonly used for monitoring Parkinson’s disease (PD), where DAT uptake amount is computed to assess PD severity. However, DAT imaging has a high cost and the risk of radiance exposure and is not available in general clinics. Recently, MRI patch of nigral region has been proposed
as a safer and easier alternative. 
This repository offers a symmetric regressor for predicting the DAT uptake amount from nigral MRI patch.
Note that, the susceptibility-map-weighted-imaging (SMWI) technique of MRI is used.
DAT uptake amount is commonly expressed as the specific binding ratio (SBR).

## Input Processing ##
From SMWI of the brain, extract right and left nigral patches where each patch encompasses a volume of `50x50x20` voxels centered at the corresponding nigrosome-1's centroid.
Pass these patches to the predictor to obtain the SBR score.

## Usage ##
`print(predict_SBR(right_nigral_patch, left_nigral_patch))`

## Dependency ##
`tensorflow 2.*`
