import os
import shutil
import cv2
import sys
import argparse
import editdistance
from WordSegmentation import wordSegmentation, prepareImg

class FilePaths:
    """ Filenames and paths to data. """
    fnCharList = '../model/charList.txt' # symbols of dictionary
    fnAccuracy = '../model/accuracy.txt' # to write accuracy of NN
    fnTrain = '../data/train/' # place to store training data
    fnWords = '../data/words/' # place/img to recognize text (test)
    fnWordsFromLines = '../data/words/fromlines_words/' # place/img to recognize text (test)
    fnLines = '../data/lines/' # place to store lines from segmented text
    fnTexts = '../data/texts/' # place to store
    fnCorpus = '../data/corpus.txt' # list of recognized words
    fnDumpRes = '../dump/results.txt'  # file, where to dump(save) the output of the NN in txt file(s)

def segment_to_words(file_path):
    """Segment line to words.
        Arguments:
                file_path - path or name of file/-s in file system.
        Returns:
                All found words from given directory.
    """
    # Get names of files in given dir to imgFiles
    if os.path.isdir(file_path):
        imgFiles = os.listdir(file_path)
    else:
        imgFiles = file_path   # If it a file - ??? (could it be?)

    found_words = []   # All found words in this dir
    for (i, f) in enumerate(imgFiles):
        print("File #", i, " Name: ", f)
        print('Segmenting words of sample %s' % f)
        # Check requirements for the image file to be processed by program
        if not check_file("%s/%s" % (file_path,f)):
            continue
        img = prepareImg(cv2.imread('%s%s' % (file_path, f)), 50)
        # execute segmentation with given parameters
        # -kernelSize: size of filter kernel (odd integer)
        # -sigma: standard deviation of Gaussian function used for filter kernel
        # -theta: approximated width/height ratio of words, filter function is distorted by this factor
        # - minArea: ignore word candidates smaller than specified area
        # Returns: List of tuples. Each tuple contains the bounding box and the image of the segmented word.
        tmp_words = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=200)
        found_words.append(tmp_words)
    return found_words


def save_tmp_data(data, path, num, dtype):
    """ Save found lines in texts
        data - array/ list of data to save
        path - dir, where to save
        num - number of text/line which words were segmented from
        dtype - type of data to save: words or lines """
    # write output to 'out/inputFileName' directory
    if not os.path.exists('%s' % path):
        os.mkdir('%s' % path)

    if dtype not in ['word', 'line']:
        raise ValueError("dtype should be in 'word' or 'line'")

    if dtype == 'word':
        # Iterate over all segmented words
        print('Segmented into %d words' % len(data))
        for (j, w) in enumerate(data):
            for (k, n) in enumerate(w):
                (wordBox, wordImg) = n
                (x, y, w, h) = wordBox  # To draw bounding box in summary image (if needed)
                fn = '%s/t%d_w%d_%d.png' % (path, num, j, k)
                cv2.imwrite(fn, wordImg)  # save word

    elif dtype == 'line':
        # iterate over all segmented lines
        print('Segmented into %d lines' % len(data))
        for (j, w) in enumerate(data):
            fn = '%s/%d%d.png' % (path, num, j)
            cv2.imwrite(fn, w)  # save line


def clear_dirs(dtype):
    """ Remove directions and their contents, that were created in previous uses of the program.
        Arguments:
                dtype - type if input data to program (which dirs do not to remove).
        Returns: -
    """
    if dtype not in ['texts', 'lines']:
        raise ValueError("dtype should be in 'texts' or 'lines'")

    if dtype == 'texts':
        shutil.rmtree(FilePaths.fnLines)
        os.mkdir('%s' % FilePaths.fnLines)

        shutil.rmtree(FilePaths.fnWords)
        os.mkdir('%s' % FilePaths.fnWords)
    elif dtype == 'lines':
        files = os.listdir(FilePaths.fnLines)
        for f in files:
            if not f.endswith(".png"):
                shutil.rmtree('%s%s' % (FilePaths.fnLines, f))

        shutil.rmtree(FilePaths.fnWords)
        os.mkdir('%s' % FilePaths.fnWords)
    print("Directions & Files Removed")

def check_file(file_path):
    """ Check requirements for the image file to be processed be program.
        Returns: true/ false - depending on whether the file is processed or not.
    """
    # Check if file is not empty/damaged
    # getsize(path) - Return the size, in bytes, of path.
    # Raise os.error if the file does not exist or is inaccessible.
    if not os.path.getsize(file_path):
        print("Warning, damaged images found: %s" % file_path)
        return False
    if not file_path.endswith(".png") or file_path.endswith(".PNG"):
        print("Warning, unsupported or invalid format of image found: %s" % file_path)
        return False
    # Image is OK
    return True
