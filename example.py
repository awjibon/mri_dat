import smwi_dat as sd
import scipy.io as io

# load the MAT file containing the nigral patches
d = io.loadmat('nigral_patch_example.mat')

# create the desired regressor model (defaule: symmetric) and predict SBR
model = sd.SMWI_DAT(model=sd.constants.model_type_symmetric)
right_sbr, left_sbr = model.predict_sbr(d['right_patch'], d['left_patch'])
print(right_sbr[0][0], left_sbr[0][0])


