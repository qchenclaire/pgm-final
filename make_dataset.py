import os, pdb, cv2
from PIL import Image
import numpy as np
import pickle
connections = 4
H, W = 28, 28
for split in ['train', 'val']:
    f = open('davis_sub_%s.txt' %split)
    lines = f.readlines()
    x = []
    y = []
    edges = []
    if connections == 4:
        for i in range(H-1):
            for j in range(W-1):
                edges.append([i*W+j, i*W+j+1])
                edges.append([i*W+j, (i+1)*W+j])
    edges = np.asarray(edges)
    for line in lines:
        print(line)
        line = line[:-1]
        image = Image.open(line + '.jpg')
        image = np.asarray(image)
        image = image/255.0
        gt = Image.open(line.replace('JPEGImage', 'Annotations') + '.png')
        gt = np.asarray(gt)//255
        unary = np.load(line+'.npy')

        w, h, c = image.shape
        image = image.reshape(w*h, c)
        gt = gt.reshape([w*h])
        unary = np.transpose(unary, (1, 2, 0))
        unary = cv2.resize(unary, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
        unary = unary.reshape(w*h, 2)
        binarys = []
        if connections == 4:
            for edge in edges:
                binary = np.exp(-(image[edge[0]]-image[edge[1]])*(image[edge[0]]-image[edge[1]])/0.5)
                binarys.append(binary)
        binarys = np.asarray(binarys)
        x.append((unary, edges, binarys))
        y.append(gt)

    pickle.dump({'X': x, 'Y':y}, open( "%s_data_davis_sub.p" %split, "wb" ) )
