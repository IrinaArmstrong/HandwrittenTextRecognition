from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from LinesSegmentation import normalize
from DataUtils import UtilsFilePaths
from UserDataLoader import FilePaths

class Sample:
    """ Single sample from the dataset
        Consist of:
            gtText - general truth text
            filePath - file path to image of word in file system
    """

    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath


class Batch:
    """ Batch (set of Samples)
        Consist of:
            gtTexts - general truth texts
            imgs - stack of images of batch
    """
    def __init__(self, gtTexts, imgs):
        # Stack images over first axis
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts


class DataLoader:
    """ Loads data, which corresponds to IAM format,
        see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

    def __init__(self, filePath, batchSize, imgSize, maxTextLen):
        """ Loader for dataset.
            - loads it at given location,
            - preprocess images and text according to parameters
            Arguments:
                filePath - path to store data. By default = ../data/
                batchSize - number of images in single batch
                imgSize - size of image. By default = (128, 32) (from Model.py file)
                maxTextLen - maximum length of text given to input of NN. By default = 32 (from Model.py file)
        """
        # Check if filePath ends with '/' (e.t. it's a dir)
        assert filePath[-1] == '/'

        self.dataAugmentation = False  # enble dataAugmentation function
        self.currIdx = 0  #
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []  # list of samples of dataset (general truth + imgFileName)

        # TODO: change paths to UtilsFilePaths or FilePaths
        # Open file with iam database word information
        f = open(filePath + 'words.txt')
        chars = set()  # list of chars, used in texts

        # bad samples = empty/wrongly separated images
        bad_samples = []  # If some images found to be damaged
        bad_samples_reference = [] # Damaged images expected
            # 'a01-117-05-02.png', 'r06-022-03-05.png']

        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue
            # strip() - deletes blank chars from the beginning and end of string, returns copy of string
            # split() - splits str into some other strings separated
            lineSplit = line.strip().split(' ')

            # it must be more then 2 substrings
            assert len(lineSplit) >= 2

            # filename: part1-part2.png transcription --> ../data/train/part1/part1-part2.png
            # Example: ../data/train/000/000-100.png
            fileNameSplit = lineSplit[0].split('-')
            # TODO: change filepath to UtilsFilePaths or FilePaths
            fileName = filePath + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '.png'

            # GT text are columns starting at 2
            gtText = self.truncateLabel(' '.join(lineSplit[1:]), maxTextLen)  # ???
            chars = chars.union(set(list(gtText)))  # Make list of chars, used in texts

            # Check if image is not empty
            if not os.path.getsize(fileName):
                bad_samples.append(lineSplit[0] + '.png')
                continue

            # Put sample into list
            self.samples.append(Sample(gtText, fileName))

        # Some images in the IAM dataset are known to be damaged, don't show warning for them
        if set(bad_samples) != set(bad_samples_reference):
            print("Warning, damaged images found:", bad_samples)
            print("Damaged images expected:", bad_samples_reference)

        # Split into training and validation set: 95% - 5%
        splitIdx = int(0.95 * len(self.samples))
        self.trainSamples = self.samples[:splitIdx]  # 114 000 samples
        self.validationSamples = self.samples[splitIdx:]  # 6 000 samples

        # Put general truth words into lists
        self.trainWords = [x.gtText for x in self.trainSamples]
        self.validationWords = [x.gtText for x in self.validationSamples]

        # Number of randomly chosen samples per epoch for training
        self.numTrainSamplesPerEpoch = 20000

        # Start with train set
        self.trainSet()

        # List of all chars in dataset
        self.charList = sorted(list(chars))


    def truncateLabel(self, text, maxTextLen):
        """ ctc_loss can't compute loss if it cannot find a mapping between text label and input labels.
            Repeat letters cost double because of the blank symbol needing to be inserted.
            If a too-long label is provided, ctc_loss returns an infinite gradient.
            So it is needed to crop text. """
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            # crop text if it is longer then maxTextLen
            if cost > maxTextLen:
                return text[:i]
        return text


    def trainSet(self):
        """ Switch to randomly chosen subset of training set. """
        self.dataAugmentation = True
        self.currIdx = 0  # reset index
        random.shuffle(self.trainSamples)  # mix samples
        self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]  # get first 25000 of them


    def validationSet(self):
        """ Switch to validation set. """
        self.dataAugmentation = False
        self.currIdx = 0  # reset index
        self.samples = self.validationSamples


    def getIteratorInfo(self):
        """ Current batch index and overall number of batches. """
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)


    def hasNext(self):
        """ True - if iterator has next element, else - False. """
        return self.currIdx + self.batchSize <= len(self.samples)


    def getNext(self):
        """ Iterator gets next batch of samples addresses, then load and preprocess images. """
        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        gtTexts = [self.samples[i].gtText for i in batchRange]
        imgs = [
            preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation)
            for i in batchRange]
        self.currIdx += self.batchSize
        return Batch(gtTexts, imgs)


def preprocess(img, imgSize, dataAugmentation=False):
    """ Put img into target img of size imgSize, transpose for TF and normalize gray-values
         img: input image,
         imgSize: image size required by programm,
         dataAugmentation: boolean, responsible for resize and rotate image for more samples or not,
    Return: processed image
    """

    # There are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])

    # Increase dataset size by applying random stretches to the images
    if dataAugmentation:
        stretch = (random.random() - 0.5)  # -0.5 .. +0.5
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1)  # random width, but at least 1
        img = cv2.resize(img, (wStretched, img.shape[0]))  # stretch horizontally by factor 0.5 .. 1.5

    # Create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  # Scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    # Transpose for TF
    img = cv2.transpose(target)

    # normalize
    return normalize(img)

