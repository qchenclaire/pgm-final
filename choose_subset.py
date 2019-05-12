import numpy as np
import pdb
length = {'train': 500, 'val': 100}
for split in ['train', 'val']:

    f = open('davis_%s.txt'%split)
    g = open('davis_sub_%s.txt'%split, 'w')
    lines = f.readlines()
    choices = np.random.choice(np.arange(len(lines)), size=length[split], replace=False)
    for i in choices:
        g.write(lines[i])
