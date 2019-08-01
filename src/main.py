import os
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
    fnTrain = '../data/' # place to store training data
    fnInfer = '../data/words/' # place/img to recognize text (test)
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
    parser.add_argument('--text', help='image - is a full text consist of many lines', action='store_true', default=True)
    parser.add_argument('--line', help='image - is line with words', action='store_true', default=False)
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
    parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
    # parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding',
    #                     action='store_true')
    parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')

    # parse_args() method actually returns some data from the options specified
    args = parser.parse_args()


    if args.text:
        imgFiles = os.listdir(FilePaths.fnTexts)
        print("Files found in data dir:{0}".format(len(imgFiles)))
        found_lines = []
        for (i, f) in enumerate(imgFiles):
            print("File #", i, " Name: ", f)
            print('Segmenting lines of sample %s' % f)
            img = cv2.imread('%s%s' % (FilePaths.fnTexts, f))
            tmp_lines = lineSegmentation(img)
            found_lines.append(tmp_lines)
            save_tmp_data(tmp_lines, FilePaths.fnLines, i, type='line')

            res_words = segment_to_words(found_lines)
            for (j, l) in enumerate(res_words):
                save_tmp_data(tmp_lines, FilePaths.fnInfer, i, type='word')


    elif args.line:
        imgFiles = os.listdir(FilePaths.fnLines)
        print("Files found in data dir:{0}".format(len(imgFiles)))


def segment_to_words(files):
    found_words = []
    for (i, f) in enumerate(files):
        print("File #", i, " Name: ", f)
        print('Segmenting words of sample %s' % f)
        img = prepareImg(cv2.imread('%s%s' % (FilePaths.fnLines, f)), 50)
        # execute segmentation with given parameters
        # -kernelSize: size of filter kernel (odd integer)
        # -sigma: standard deviation of Gaussian function used for filter kernel
        # -theta: approximated width/height ratio of words, filter function is distorted by this factor
        # - minArea: ignore word candidates smaller than specified area
        tmp_words = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
        found_words.append(tmp_words)
    return found_words


def save_tmp_data(data, path, num, type):
    """ Save found lines in texts """
    # write output to 'out/inputFileName' directory
    if not os.path.exists('%s' % path):
        os.mkdir('%s' % path)

    if not type in ['word', 'line']:
        raise ValueError("Type should be in 'word' or 'line'")
    if type == 'word':
        # iterate over all segmented words
        print('Segmented into %d words' % len(data))
        for (j, w) in enumerate(data):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            cv2.imwrite('%s/%d%d.png' % (data, num, j), wordImg)  # save word

    elif type == 'line':
        # iterate over all segmented lines
        print('Segmented into %d lines' % len(data))
        for (j, w) in enumerate(data):
            # cv2.imshow('line', w)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite('%s/%d%d.png' % (path, num, j), w)  # save line

if __name__ == '__main__':
	main()