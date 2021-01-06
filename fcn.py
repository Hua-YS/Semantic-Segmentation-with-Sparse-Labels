from backbone import *
import tensorflow as tf

def fcn_festa(patch_size, test=True, noclutter=True):

    nb_classes = 5 if noclutter else 6
    base_model = VGG16(patch_size, True)
    x4 = base_model.get_layer('block4_pool').output
    x4 = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x4_up')(x4)
    x5 = base_model.get_layer('block5_pool').output
    x5 = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x5_up')(x5)

    x = Add(name='final_feat')([x4, x5])
    x_out =  Conv2D(nb_classes, (1, 1), activation='softmax', padding='same', name='final_out')(x)

    if test == True:
        return Model(base_model.inputs, x_out, name='vgg16')#fcn_festa')
    
    return Model(base_model.inputs, [x, x_out], name='vgg16')#fcn_festa')

