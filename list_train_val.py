import os
train_file = 'ImageSets/2017/train.txt'
val_file = 'ImageSets/2017/val.txt'
out_train = 'davis_train.txt'
out_val = 'davis_val.txt'
import pdb
f = open(train_file, 'r')
g = open(out_train, 'w')
sequences = f.readlines()
for sequence in sequences:
    sequence = sequence[:-1]
    image_path = os.path.join('DAVIS', sequence, 'JPEGImage')
    images = os.listdir(image_path)
    for image in images:
        g.write(os.path.join(image_path, image.split('.')[0]))
        g.write('\n')
f.close()
g.close()

f = open(val_file, 'r')
g = open(out_val, 'w')
sequences = f.readlines()
for sequence in sequences:
    sequence = sequence[:-1]
    image_path = os.path.join('DAVIS', sequence, 'JPEGImage')
    images = os.listdir(image_path)
    for image in images:
        g.write(os.path.join(image_path, image.split('.')[0]))
        g.write('\n')
f.close()
g.close()
