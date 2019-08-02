import os
import shutil
import cv2
import sys
import argparse
import editdistance
from LinesSegmentation import lineSegmentation
from WordSegmentation import wordSegmentation, prepareImg
# from DataLoader import DataLoader, Batch
# from Model import Model, DecoderType
# from SamplePreprocessor import preprocess

class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt' # symbols of dictionary
    fnAccuracy = '../model/accuracy.txt' # to write accuracy of NN
    fnTrain = '../data/train' # place to store training data
    fnWords = '../data/words/' # place/img to recognize text (test)
    fnLines = '../data/lines/' # place to store lines from segmented text
    fnTexts = '../data/texts/' # place to store
    fnCorpus = '../data/corpus.txt' # list of recognized words


def main():
    "Main function: parse arguments"

    # Parse optional command line args
    parser = argparse.ArgumentParser()
    # add_argument() - specify which command-line options the program is willing to accept.
    # action - function, that will be executed, when appropriate argument received
    # store_true - option, if appropriate argument received, then = true
    parser.add_argument('--text', help='image - is a full text consist of many lines', action='store_true') #, default=True
    parser.add_argument('--line', help='image - is line with words', action='store_true') #, default=False
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
    parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
    # parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding',
    #                     action='store_true')
    parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')

    # parse_args() method actually returns some data from the options specified
    args = parser.parse_args()


    if args.text:
        clear_dirs("texts")

        imgFiles = os.listdir(FilePaths.fnTexts)
        print("Files found in data dir:{0}".format(len(imgFiles)))
        # found_lines = [] # Lines found in ALL texts
        # i - text index, f - filename of text image
        for (i, f) in enumerate(imgFiles):
            print("File #", i, " Name: ", f)
            print('Segmenting lines of sample %s' % f)
            img = cv2.imread('%s%s' % (FilePaths.fnTexts, f))
            tmp_lines = lineSegmentation(img)  # Lines found in one text
            # found_lines.append(tmp_lines)
            fpath = ("%s/text%d_lines/" % (FilePaths.fnLines, i))
            save_tmp_data(tmp_lines, fpath, i, dtype='line')

            # res_words - list of tuples
            res_words = segment_to_words(fpath)
            wfpath = ("%s/text%d_words/" % (FilePaths.fnWords, i))
            save_tmp_data(res_words, wfpath, i, dtype='word')

    elif args.line:
        clear_dirs("lines")

        imgFiles = os.listdir(FilePaths.fnLines)
        print("Files found in data dir:  {0}".format(len(imgFiles)))

        # res_words - list of tuples
        res_words = segment_to_words(FilePaths.fnLines)
        wfpath = ("%s/fromlines_words/" % FilePaths.fnWords)
        save_tmp_data(res_words, wfpath, 0, dtype='word')

    # TODO: Create model


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
        # os.mkdir('%s' % (path, ))
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
            # cv2.imshow('line', w)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
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


if __name__ == '__main__':
	main()