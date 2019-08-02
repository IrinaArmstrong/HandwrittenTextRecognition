import os
import shutil
import cv2
import sys
import argparse
import editdistance
from LinesSegmentation import lineSegmentation
from WordSegmentation import wordSegmentation, prepareImg
from UserDataLoader import segment_to_words, save_tmp_data, clear_dirs, FilePaths, check_file
# from DataLoader import DataLoader, Batch
# from Model import Model, DecoderType
# from SamplePreprocessor import preprocess


def main():
    """ Main function:
            - parse arguments
            - created model of NN
    """

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
        # Check if dir with data is not empty
        imgFiles = os.listdir(FilePaths.fnTexts)
        if not imgFiles:
            print("Files found in data dir: %s" % FilePaths.fnTexts)
            return -1
        print("Error! No files found in data dir:{0}".format(len(imgFiles)))
        # found_lines = [] # Lines found in ALL texts
        # i - text index, f - filename of text image
        for (i, f) in enumerate(imgFiles):
            print("File #", i, " Name: ", f)
            print('Segmenting lines of sample %s' % f)
            # Check requirements for the image file to be processed be program
            if not check_file(f):
                continue
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





if __name__ == '__main__':
	main()