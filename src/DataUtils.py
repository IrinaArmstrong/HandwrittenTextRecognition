from __future__ import division
from __future__ import print_function

import os
import re
import random
import numpy as np
import cv2

from UserDataLoader import FilePaths

"""
File contains functions for structuring the system of training data folders, 
formatting the data itself, and other useful actions to bring training data 
to the format required for neural network training.

NOTE:
* Dir structure should looks like this:
 data
    --words.txt
    --train
        ----000
        ------000-0000.png
        ------000-0001.png
        --------...
        ------000-1500.png
        ----001
        ----...
        ----007
        
* Each filename should consist of: 
    000-0001.png
        000 - name of data folder, depends on font of the texts used to generate this sample;
        0001 - number of this sample, should be four-digit number;
        .png - extension of this file, should always be .png/.PNG.
"""

def createDiscriptionFile (path) :
    subdirs = os.listdir(path)
    for num, sb in enumerate(subdirs):
        renameFiles("%s%s" % (path,sb))


def renameFiles(dir):
    count = 0
    files = os.listdir(dir)
    for num, f in enumerate(files):
        match = re.search(r'\d\d\d\D\d*\.png', f)
        if  match:
            continue
        elif f.endswith(".txt"):
            continue
        else:
            os.rename("%s/%s" % (dir, f), "%s/%s-%s" % (dir, dir.split('/')[3], f))
            print("Renamed from %s/%s to %s/%s-%s" % (dir, f, dir, dir.split('/')[3], f))
            count = count+1
    print("Total, renamed: %d files in dir:%s" % (count, dir))

if __name__ == '__main__':
	createDiscriptionFile(FilePaths.fnTrain)