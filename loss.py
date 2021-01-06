#from keras.objectives import *
#from keras.metrics import binary_crossentropy
import keras.backend as K
import tensorflow as tf #tensorflow.compat.v1 as tf

def L_festa(_, y_pred):

    alpha = 0.5 # weight of neighbour in the feature space
    beta = 1.5 # weight of neighbour in the image space
    gamma = 1 # weight of far-away in the feature space

    sample_ratio = 0.01 # measure only sample_ratio % samples for computational efficiency

    _, h, w, c = K.int_shape(y_pred)
    batch_size = K.shape(y_pred)[0]
    # randomly sampling sample_ratio % feature samples
    y_pred_all_reshape = K.reshape(y_pred, (batch_size, -1, c))
    random_idxs = tf.random_shuffle(tf.range((h-2)*(w-2)))[:int(h*w*sample_ratio)]
    random_idxs = random_idxs + 257
    y_pred_reshape = tf.gather(y_pred_all_reshape, random_idxs, axis=1)

    # ***************************** cosine similarity ***************************
    # calculating distance in the feature space
    xixj = tf.matmul(y_pred_reshape, tf.transpose(y_pred_all_reshape, [0, 2, 1]))
    similarity = xixj/(tf.expand_dims(tf.norm(y_pred_reshape, axis=-1), axis = -1)*tf.expand_dims(tf.norm(y_pred_all_reshape, axis=-1), axis = 1)+1e-8)
    faraway_feature = tf.reduce_min(similarity, axis = -1) # feature with minimum similarity in the feaure space

    # ***************************** euclidean distance ***************************
    distance = tf.expand_dims(tf.square(tf.norm(y_pred_reshape, axis=-1)), axis=-1) - 2*xixj + tf.expand_dims(tf.square(tf.norm(y_pred_all_reshape, axis=-1)), axis = 1)
    
    ind_diag = K.cast(tf.stack([tf.range(int(h*w*sample_ratio)), random_idxs], axis=1), 'int64')
    no_diag = tf.sparse_to_dense(ind_diag, [int(h*w*sample_ratio), h*w], K.repeat_elements(tf.constant([1.0]), int(h*w*sample_ratio), 0), validate_indices=False)*(tf.reduce_max(distance)+1)
    no_diag = tf.tile(K.flatten(no_diag), (batch_size, ))
    no_diag = K.reshape(no_diag, (batch_size, int(h*w*sample_ratio), h*w))
    
    neighbour_feature = tf.reduce_min(distance+no_diag, axis = -1) # feature with minimum distance in the feature space
    
    # get indexes of 8-neighbouring pixels of the center pixel
    random_idxs_L = random_idxs - 1
    random_idxs_R = random_idxs + 1
    random_idxs_TL = random_idxs - h -1
    random_idxs_T = random_idxs - h
    random_idxs_TR = random_idxs - h + 1
    random_idxs_BL = random_idxs + h -1
    random_idxs_B = random_idxs + h
    random_idxs_BR = random_idxs + h + 1

    ind_L = K.cast(tf.stack([tf.range(int(h*w*sample_ratio)), random_idxs_L], axis=1), 'int64')
    ind_R = K.cast(tf.stack([tf.range(int(h*w*sample_ratio)), random_idxs_R], axis=1), 'int64')
    ind_TL = K.cast(tf.stack([tf.range(int(h*w*sample_ratio)), random_idxs_TL], axis=1), 'int64')
    ind_T = K.cast(tf.stack([tf.range(int(h*w*sample_ratio)), random_idxs_T], axis=1), 'int64')
    ind_TR = K.cast(tf.stack([tf.range(int(h*w*sample_ratio)), random_idxs_TR], axis=1), 'int64')
    ind_BL = K.cast(tf.stack([tf.range(int(h*w*sample_ratio)), random_idxs_BL], axis=1), 'int64')
    ind_B = K.cast(tf.stack([tf.range(int(h*w*sample_ratio)), random_idxs_B], axis=1), 'int64')
    ind_BR = K.cast(tf.stack([tf.range(int(h*w*sample_ratio)), random_idxs_BR], axis=1), 'int64')
    ind = tf.concat([ind_L, ind_R, ind_TL, ind_T, ind_TR, ind_BL, ind_B, ind_BR], axis=0)
    mask = tf.sparse_to_dense(ind, [int(h*w*sample_ratio), h*w], K.repeat_elements(tf.constant([1.0]), int(h*w*sample_ratio)*8, 0), validate_indices=False)
    distance_mask = tf.multiply(distance+no_diag, mask) # calculate distances between 8-neighbouring pixels and the center pixel
    neighbour_spatial = tf.reduce_min(distance_mask, axis = -1) # feature with minimum distance in the image space
    
    delta = alpha*neighbour_feature++beta*neighbour_spatial+gamma*faraway_feature

    loss_reg = tf.reduce_mean(delta)
    return loss_reg

