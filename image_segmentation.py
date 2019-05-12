"""
===========================================
Semantic Image Segmentation on Pascal VOC
===========================================
This example demonstrates learning a superpixel CRF
for semantic image segmentation.
To run the experiment, please download the pre-processed data from:
http://www.ais.uni-bonn.de/deep_learning/downloads.html

The data consists of superpixels, unary potentials, and the connectivity
structure of the superpixels.
The unary potentials were originally provided by Philipp Kraehenbuehl:
http://graphics.stanford.edu/projects/densecrf/

The superpixels were extracted using SLIC.
The code for generating the connectivity graph and edge features will be made
available soon.

This example does not contain the proper evaluation on pixel level, as that
would need the Pascal VOC 2010 dataset.
"""
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
import pdb, time, os
from pystruct import learners
import pystruct.models as crfs
from pystruct.utils import SaveLogger
output_folder = 'out'
method = 'max-product'
evaluate_val = True
import cv2

data_train = pickle.load(open("train_data_davis_sub.p", "rb"))
C = 0.01
n_states = 2
print("number of samples: %s" % len(data_train['X']))

model = crfs.EdgeFeatureGraphCRF(inference_method=method)

experiment_name = "edge_features_one_slack_trainval_%f" % C

ssvm = learners.NSlackSSVM(
    model, verbose=2, C=C, max_iter=25, n_jobs=-1,
    tol=0.0001, show_loss_every=5,
    logger=SaveLogger(experiment_name + ".pickle", save_every=5),
    inactive_threshold=1e-3, inactive_window=10, batch_size=100)
start = time.time()
ssvm.fit(data_train['X'], data_train['Y'])
print('average training time: %f s/iter'%((time.time()-start)/25.0))
data_val = pickle.load(open("val_data_davis_sub.p", "rb"))
start = time.time()
if evaluate_val:
    y_pred = ssvm.predict(data_val['X'])
else:
    y_pred = ssvm.predict(data_train['X'])

print('average training time: %f s/im'%((time.time()-start)/100.0))
if evaluate_val:
    f = open('davis_sub_val.txt')
else:
    f = open('davis_sub_train.txt')
lines = f.readlines()
for i, pred in enumerate(y_pred):
    filename = (lines[i][:-1]+'.png').replace('JPEGImage', method).replace('DAVIS', output_folder)
    paths = filename.split('/')[:-1]
    parent_folder = '/'.join(paths)
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    cv2.imwrite(filename, 255 * pred.reshape(28, 28))
# we throw away void superpixels and flatten everything
if evaluate_val:
    y_pred, y_true = np.hstack(y_pred), np.hstack(data_val['Y'])
else:
    y_pred, y_true = np.hstack(y_pred), np.hstack(data_train['Y'])
y_pred = y_pred[y_true != 255]
y_true = y_true[y_true != 255]

print("Score on validation set: %f" % np.mean(y_true == y_pred))
