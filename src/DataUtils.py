from __future__ import division
from __future__ import print_function

import os
import re
import io
import random
import numpy as np
import cv2


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

class UtilsFilePaths:
    fnTrain = '../data/train/' # place to store training data
    fnInfo = '../data/train/words.txt'  # file, where to write description of training dataset


def main():
    write_infofile = True
    rename = False
    subdirs = os.listdir(UtilsFilePaths.fnTrain)

    if rename:
        for num, sb in enumerate(subdirs):
            renameFiles("%s%s" % (UtilsFilePaths.fnTrain, sb))

    if write_infofile:
        createDiscriptionFile(UtilsFilePaths.fnTrain, UtilsFilePaths.fnInfo)


def createDiscriptionFile (dirpath, fpath) :

    title = """*
* format: 000-0.png текст
*
*     000     -> number of subdir in train data direction, where this word located 
*               (depends on type of font this word was written);
*     0       -> id of word in the subdir, should be from 0 to 7999;
*     .png    -> extension of this word, should always be .png/.PNG;
*     текст   -> the transcription for this word
*
"""
    strs_to_write = []
    subdirs = os.listdir(dirpath)
    words_txt = _searchTxt(subdirs)

    # Remove info file (words.txt)
    if words_txt != 'None':
        subdirs.remove(words_txt)
    else :
        print("No files with .txt extension found in direction: %s" % dirpath)

    for num, sub in enumerate(subdirs):
        # Subdir prefix
        subdir_str = "%s-" % sub

        files = os.listdir("%s%s" % (dirpath, sub))
        # Search for .txt file with info about this subdir
        file = _searchTxt(files)
        if not file:
            print("No files with .txt extension found in direction: %s" % sub)

        file_path = "%s%s/%s" % (dirpath, sub, file)

        # Open info file of sundir
        file_handler =  io.open(file_path, 'r', encoding='utf-8')
        # String, containing all words in this subdirectory
        str_to_write = ""
        while True:
            # Get next line from file
            line = file_handler.readline()\
            # .strip()
            # If line is empty then end of file reached
            if not line:
                break
            str_to_write += "%s%s" % (subdir_str, line)

        # Append string to list of strings
        strs_to_write.append(str_to_write)
        # Close Close
        file_handler.close()

    # If file do not exist create it, then write list of strings about files in subdirs in it
    with open(fpath, 'w+') as f:
        f.write(title)
        for s in strs_to_write:
            f.write("%s" % s)
            print("%s" % s)
    print("Ended")


def _searchTxt(files):
    """ Search for .txt file with info about subdir. """
    for num, f in enumerate(files):
        if f.endswith(".txt"):
            return f
    return 'None'

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
    main()