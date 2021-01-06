import numpy as np
import cv2
import os
import scipy.io as sio
from sklearn.metrics import confusion_matrix

folder_path = './data/Vaihingen/'
im_header = 'top_mosaic_09cm_area'
trainval_set = [1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37]
test_set = [11, 15, 28, 30, 34]
im_path = folder_path + 'img/'
gt_path = folder_path + 'eroded_gt/mask_' # for calculating scores
eps = 1e-14

def dataloader(patch_size=256, stride_size=64, an_type='polygon', an_id=1, noclutter=True, remove_null=True):

    # path of sparse label
    sparse_label_path = folder_path + an_type + '/an' + str(an_id) + '/mask_'

    # crop images to patches
    for fid in range(len(trainval_set)):
        print(im_header + str(trainval_set[fid]) + '.png')
        X, y = img2patch(im_header + str(trainval_set[fid]) + '.png', sparse_label_path, patch_size, stride_size, noclutter, remove_null)    
        X_tra = np.concatenate([X_tra, X], axis=0) if fid>0 else X
        y_tra = np.concatenate([y_tra, y], axis=0) if fid>0 else y

    for fid in range(len(test_set)):
        print(im_header + str(test_set[fid]) + '.tif')
        X, y = img2patch(im_header + str(test_set[fid]) + '.tif', gt_path, patch_size, stride_size, noclutter, remove_null)
        X_test = np.concatenate([X_test, X], axis=0) if fid>0 else X
        y_test = np.concatenate([y_test, y], axis=0) if fid>0 else y

    X_tra = np.float32(X_tra)
    y_tra = np.uint8(y_tra)
    X_test = np.float32(X_test)
    y_test = np.uint8(y_test)
    print('the size of training data:', np.shape(X_tra))

    return X_tra, y_tra, X_test, y_test


def img2patch(filename, label_path, patch_size=256, stride_size=256, noclutter=True, remove_null=True):

    im = cv2.imread(im_path + filename[:-4] + '.tif')
    gt = bgr2index(cv2.imread(label_path + filename))
    gt = gt[:, :, 0:5] if noclutter else gt
    
    # crop an image/mask to patches
    X, y = [], []
    im_row, im_col, _ = np.shape(im)
    steps_row = int(np.floor((im_row - (patch_size - stride_size)) / stride_size))
    steps_col = int(np.floor((im_col - (patch_size - stride_size)) / stride_size))

    for i in range(steps_row+1):
        for j in range(steps_col+1):
            if i == steps_row:
                if j == steps_col:
                    X_patch = im[-patch_size:im_row, -patch_size:im_col, :]
                    y_patch = gt[-patch_size:im_row, -patch_size:im_col, :]
                else:
                    X_patch = im[-patch_size:im_row, (j * stride_size):(j * stride_size + patch_size),:]
                    y_patch = gt[-patch_size:im_row, (j * stride_size):(j * stride_size + patch_size),:]
            else:
                if j == steps_col:
                    X_patch = im[(i * stride_size):(i * stride_size + patch_size), -patch_size:im_col, :]
                    y_patch = gt[(i * stride_size):(i * stride_size + patch_size), -patch_size:im_col, :]
                else:
                    X_patch = im[(i * stride_size):(i * stride_size + patch_size), (j * stride_size):(j * stride_size + patch_size), :]
                    y_patch = gt[(i * stride_size):(i * stride_size + patch_size), (j * stride_size):(j * stride_size + patch_size), :]
            
            if remove_null and np.sum(y_patch) == 0:
                continue

            X.append(X_patch)
            y.append(y_patch)

    X = np.float32(X)
    y = np.uint8(y)
    return X, y

def bgr2index(gt_bgr, eroded=False):
    # mapping BGR W x H x 3 image to W x H x C class index
    # opencv read image to BGR format
    im_col, im_row, _ = np.shape(gt_bgr)
    gt = np.zeros((im_col, im_row, 6)) if not eroded else np.zeros((im_col, im_row, 7))
    gt[(gt_bgr[:, :, 2] == 255) & (gt_bgr[:, :, 1] == 255) & (gt_bgr[:, :, 0] == 255), 0] = 1
    gt[(gt_bgr[:, :, 2] == 0) & (gt_bgr[:, :, 1] == 0) & (gt_bgr[:, :, 0] == 255), 1] = 1
    gt[(gt_bgr[:, :, 2] == 0) & (gt_bgr[:, :, 1] == 255) & (gt_bgr[:, :, 0] == 255), 2] = 1
    gt[(gt_bgr[:, :, 2] == 0) & (gt_bgr[:, :, 1] == 255) & (gt_bgr[:, :, 0] == 0), 3] = 1
    gt[(gt_bgr[:, :, 2] == 255) & (gt_bgr[:, :, 1] == 255) & (gt_bgr[:, :, 0] == 0), 4] = 1
    gt[(gt_bgr[:, :, 2] == 255) & (gt_bgr[:, :, 1] == 0) & (gt_bgr[:, :, 0] == 0), 5] = 1
    if eroded:
        gt[(gt_bgr[:, :, 2] == 0) & (gt_bgr[:, :, 1] == 0) & (gt_bgr[:, :, 0] == 0), 6] = 1

    return gt

def index2bgr(c_map, bgr=True):

    # mapping W x H x 1 class index to W x H x 3 BGR image
    im_col, im_row = np.shape(c_map)
    c_map_r = np.zeros((im_col, im_row), 'uint8')
    c_map_g = np.zeros((im_col, im_row), 'uint8')
    c_map_b = np.zeros((im_col, im_row), 'uint8')
    c_map_r[c_map == 0] = 255
    c_map_r[c_map == 1] = 0
    c_map_r[c_map == 2] = 0
    c_map_r[c_map == 3] = 0
    c_map_r[c_map == 4] = 255
    c_map_r[c_map == 5] = 255
    c_map_g[c_map == 0] = 255
    c_map_g[c_map == 1] = 0
    c_map_g[c_map == 2] = 255
    c_map_g[c_map == 3] = 255
    c_map_g[c_map == 4] = 255
    c_map_g[c_map == 5] = 0
    c_map_b[c_map == 0] = 255
    c_map_b[c_map == 1] = 255
    c_map_b[c_map == 2] = 255
    c_map_b[c_map == 3] = 0
    c_map_b[c_map == 4] = 0
    c_map_b[c_map == 5] = 0
    c_map_rgb = np.zeros((im_col, im_row, 3), 'uint8');
    c_map_rgb[:, :, 0] = c_map_b if bgr else c_map_r
    c_map_rgb[:, :, 1] = c_map_g
    c_map_rgb[:, :, 2] = c_map_r if bgr else c_map_b
    
    return c_map_rgb



def eval_image(gt, pred, acc1, acc2, acc3, acc4, acc5, noclutter=True):

    im_row, im_col = np.shape(pred)
    cal_classes = 5 if noclutter else 6 # no. of classes to calculate scores

    if noclutter:
        gt[gt == 5] = 6 # pixels in clutter are not considered (regarding them as boundary)

    pred[gt == 6] = 6 # pixels on the boundary are not considered for calculating scores
    OA = np.float32(len(np.where((np.float32(pred) - np.float32(gt)) == 0)[0])-len(np.where(gt==6)[0]))/np.float32(im_col*im_row-len(np.where(gt==6)[0]))
    acc1 = acc1 + len(np.where((np.float32(pred) - np.float32(gt)) == 0)[0])-len(np.where(gt==6)[0])
    acc2 = acc2 + im_col*im_row-len(np.where(gt==6)[0])
    pred1 = np.reshape(pred, (-1, 1))
    gt1 = np.reshape(gt, (-1, 1))
    idx = np.where(gt1==6)[0]
    pred1 = np.delete(pred1, idx)
    gt1 = np.delete(gt1, idx)
    CM = confusion_matrix(pred1, gt1)
    for i in range(cal_classes):
        tp = np.float32(CM[i, i])
        acc3[i] = acc3[i] + tp
        fp = np.sum(CM[:, i])-tp
        acc4[i] = acc4[i] + fp
        fn = np.sum(CM[i, :])-tp
        acc5[i] = acc5[i] + fn
        P = tp/(tp+fp+eps)
        R = tp/(tp+fn+eps)
        f1 = 2*(P*R)/(P+R+eps)

    return acc1, acc2, acc3, acc4, acc5


def pred_image(filename, model, patch_size, stride_size):

    # croppping an image into patches for prediction    
    X, _ = img2patch(filename, gt_path, patch_size, stride_size, True, False)
    pred_patches = model.predict(X)

    # rearranging patchess into an image
    # For pixels with multiple predictions, we take their averages
    im_row, im_col, _ = np.shape(cv2.imread(im_path + filename))
    steps_col = int(np.floor((im_col - (patch_size - stride_size)) / stride_size))
    steps_row = int(np.floor((im_row - (patch_size - stride_size)) / stride_size))
    im_out = np.zeros((im_row, im_col, np.shape(pred_patches)[-1]))
    im_index = np.zeros((im_row, im_col, np.shape(pred_patches)[-1])) # counting the number of predictions for each pixel

    patch_id = 0
    for i in range(steps_row+1):
        for j in range(steps_col+1):
            if i == steps_row:        
                if j == steps_col:        
                    im_out[-patch_size:im_row, -patch_size:im_col, :] += pred_patches[patch_id]
                    im_index[-patch_size:im_row, -patch_size:im_col, :] += np.ones((patch_size, patch_size, np.shape(pred_patches)[-1]))
                else:
                    im_out[-patch_size:im_row, (j * stride_size):(j * stride_size + patch_size), :] += pred_patches[patch_id]
                    im_index[-patch_size:im_row, (j * stride_size):(j * stride_size + patch_size), :] += np.ones((patch_size, patch_size, np.shape(pred_patches)[-1]))
            else:
                if j == steps_col:
                    im_out[(i * stride_size):(i * stride_size + patch_size), -patch_size:im_col, :] += pred_patches[patch_id]
                    im_index[(i * stride_size):(i * stride_size + patch_size), -patch_size:im_col, :] += np.ones((patch_size, patch_size, np.shape(pred_patches)[-1]))
                else:
                    im_out[(i * stride_size):(i * stride_size + patch_size), (j * stride_size):(j * stride_size + patch_size), :] += pred_patches[patch_id]
                    im_index[(i * stride_size):(i * stride_size + patch_size), (j * stride_size):(j * stride_size + patch_size), :] += np.ones((patch_size, patch_size, np.shape(pred_patches)[-1]))
            patch_id += 1

    return im_out/im_index

def TestModel(model, output_folder='model', patch_size=256, stride_size=128, noclutter=True):
    
    # path for saving output
    output_path = folder_path + 'outputs/' + output_folder + '/'
    if not os.path.isdir(output_path):
        print('The target folder is created.')
        os.mkdir(output_path)

    nb_classes = 5 if noclutter else 6
    acc1 = 0.0 # accumulator for correctly classified pixels
    acc2 = 0.0 # accumulator for all valid pixels (not including label 0 and 6)
    acc3 = np.zeros((nb_classes, 1)) # accumulator for true positives
    acc4 = np.zeros((nb_classes, 1)) # accumulator for false positives
    acc5 = np.zeros((nb_classes, 1)) # accumulator for false negatives

    # predicting and measuring all images
    for im_id in range(len(test_set)):
        filename = im_header + str(test_set[im_id]) + '.tif'
        print(im_id+1, '/', len(test_set), ': predicting ', filename)
        gt = bgr2index(cv2.imread(gt_path + filename), True)

        # predict one image
        pred = pred_image(filename, model, patch_size, stride_size)
        pred = np.argmax(pred, -1)
        gt = np.argmax(gt, -1)

        # evaluate one image
        acc1, acc2, acc3, acc4, acc5 = eval_image(gt, pred, acc1, acc2, acc3, acc4, acc5, noclutter)
        cv2.imwrite(output_path+filename, index2bgr(pred, True))
        print('Prediction is done. The output is saved in ', output_path)

    OA = acc1/acc2

    f1 = np.zeros((nb_classes, 1));
    iou = np.zeros((nb_classes, 1));
    #ca = np.zeros((nb_classes, 1));
    for i in range(nb_classes):
        P = acc3[i]/(acc3[i]+acc4[i])
        R = acc3[i]/(acc3[i]+acc5[i])
        f1[i] = 2*(P*R)/(P+R)
        iou[i] = acc3[i]/(acc3[i]+acc4[i]+acc5[i])
        #ca[i] =  acc3[i]/(acc3[i]+acc4[i])

    f1_mean = np.mean(f1)
    iou_mean = np.mean(iou)
    #ca_mean = np.mean(ca)
    print('mean f1:', f1_mean, '\nmean iou:', iou_mean, '\nOA:', OA)

    return 'All predicitions are done, and output images are saved.'

