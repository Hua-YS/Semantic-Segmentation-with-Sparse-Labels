from fcn import *
from utils import *
from loss import *
from keras.optimizers import Nadam
from keras.callbacks import ReduceLROnPlateau

# **************************** path ********************************
weight_path = 'weights/fcn.h5'

# ******************** image & label config ***********************
patch_size = 256 # size of each patch
stride_size = 64 # stride of sliding window
noclutter = True # whether taking cluuter/background into consideration
an_type = 'polygon' # type of sparse annotations
an_id = 1 # id of annotators: 1 and 2 are expert, 3 and 4 are non-expert

# ************************ training scheme *************************
batch_size = 5 # size of training batch
epochs = 100 # number of training epochs
lr = 2e-4 # initial learning rate
lambda_festa = 0.1 # lambda in Eq. 2, weight of festa
remove_null = True # whether removing patches have no sparse annotations
loss = [L_festa, 'categorical_crossentropy'] # final loss Eq. 2
loss_weights = [lambda_festa, 1] # weight of each loss term in Eq. 2

# ********************** loading data *****************************
print('loading training data ...')
X_tra, y_tra, _, _ = dataloader(patch_size, stride_size, an_type, an_id, noclutter, remove_null)
print('training data is loaded.')
# ********************* initialize model ********************
model = fcn_festa(patch_size, False, noclutter)
optimizer = Nadam(lr=lr) # define yourself, e.g. sgd, adam
model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=['accuracy'])
print('model is built')

# ********************* train ***********************************
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=0, min_lr=0.5e-10)
model.fit(X_tra, [y_tra, y_tra], batch_size=batch_size, shuffle = True, epochs=epochs, validation_split=0.05, callbacks=[lr_reducer])
model.save_weights(weight_path)


