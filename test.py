from fcn import *
from utils import *

# ************************* path ****************************
weight_path = 'weights/fcn_line.h5' # weights trained on line labels create by annotator 1
out_folder = 'festa'

# ******************* image configuration *******************
patch_size = 256 # size of each patch
stride_size = 128 # stride of sliding window
noclutter = True

# ********************* initialize model ********************
model = fcn_festa(patch_size, True, noclutter)
model.load_weights(weight_path, by_name=True)

# ********************* evaluate ****************************
TestModel(model, out_folder, patch_size, stride_size, noclutter)

