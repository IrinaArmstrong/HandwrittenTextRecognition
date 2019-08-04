import os
import shutil
import cv2
import sys
import argparse
import editdistance
from LinesSegmentation import lineSegmentation
from WordSegmentation import wordSegmentation, prepareImg
from UserDataLoader import segment_to_words, save_tmp_data, clear_dirs, FilePaths, check_file
from TrainDataLoader import DataLoader, Batch, preprocess
from Model import Model, DecoderType, ModelFilePaths


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
            print("Error! No files found in data dir:%s" % FilePaths.fnTexts)
            return -1
        print("Files found in data dir:{0}".format(len(imgFiles)))
        # found_lines = [] # Lines found in ALL texts
        # i - text index, f - filename of text image
        for (i, f) in enumerate(imgFiles):
            print("File #", i, " Name: ", f)
            print('Segmenting lines of sample %s' % f)
            # Check requirements for the image file to be processed be program
            if not check_file("%s/%s" % (FilePaths.fnTexts, f)):
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

    # Set decoder type
    decoderType = DecoderType.BestPath  # bestPath -> default decoder
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    # elif args.wordbeamsearch:
    #     decoderType = DecoderType.WordBeamSearch

    # Train or validate on IAM dataset
    if args.train or args.validate:
        # Load training data, create TF model
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

        # Save characters of model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

        # Save words contained in dataset into file
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

        # Execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True)
            validate(model, loader)

        # Infer text on test images
    else:
        if os.path.exists('%s' % FilePaths.fnAccuracy):
            print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)

        if args.text:
            dirs = os.listdir(FilePaths.fnWords)
            if not dirs:
                print("No directions with words found in data direction: %s" % FilePaths.fnWords)
                return -1
            # i - text index, f - filename of text image
            for (i, d) in enumerate(dirs):
                print("Direction from text #", i, " Named: ", d)
                print('Executing files from %s...' % d)
                files = os.listdir("%s%s/" % (FilePaths.fnWords, d))
                if not files:
                    print("No words found in data direction: %s" % ("%s%s/" % (FilePaths.fnWords, d)))
                    continue
                infer(model, "%s%s/" % (FilePaths.fnWords, d))

        elif args.line:
            files = os.listdir(FilePaths.fnWordsFromLines)
            if not files:
                print("No files with words found in data direction: %s" % FilePaths.fnWordsFromLines)
                return -1
        infer(model, FilePaths.fnWordsFromLines)


def train(model, loader):
    """ Train NN.
        Arguments:
            model - constructed model of NN,
            loader - TrainDataLoader type object to load training data.
    """
    epoch = 0  # number of training epochs since start
    bestCharErrorRate = float('inf')  # best validation character error rate
    noImprovementSince = 0  # number of epochs no improvement of character error rate occurred
    earlyStopping = 5  # stop training after this number of epochs without improvement

    # Endless cycle for training, only ends when no improvement of character
    # error rate occurred more then number of epochs, chosen for early stopping
    while True:
        # Count epochs
        epoch += 1
        print('Epoch:', epoch)

        # Train
        print('Train NN')
        # Load train set (of 25000 images = 1 epoch)
        loader.trainSet()

        while loader.hasNext():
            # Get current batch index and overall number of batches
            iterInfo = loader.getIteratorInfo()
            # iterator gets next batch samples addresses, then load and preprocess images
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        # Validate
        charErrorRate = validate(model, loader)

        # If better validation accuracy achieved, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved: %f%%, save model...' % (charErrorRate * 100.0))
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0  # Reset counter
            model.save()  # Save new snapshot of model
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (charErrorRate * 100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1  # Increment counter

        # Stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break


def validate(model, loader):
    """ Validate NN
        Arguments:
            model - constructed model of NN,
            loader - TrainDataLoader type object to load training data.
        Return:
            charErrorRate - percentage ratio of errors on single characters.
    """
    print('Validate NN')
    # Switch DataLoader to validation set
    loader.validationSet()
    numCharErr = 0  # error rate for chars recognized wrongly
    numCharTotal = 0  # counter for chats total
    numWordOK = 0  # counter for words recognized rightly
    numWordTotal = 0  # counter for words total
    while loader.hasNext():
        # Print info about training process
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0], '/', iterInfo[1])
        # Get new batch
        batch = loader.getNext()
        # Feed a batch into the NN to recognize the texts
        # Returns: (texts, probs)
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            # TODO: Make able to recognize errors regardless of case upper/lower
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            #  edit distance - measure of distinction between two words
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # Print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate * 100.0, wordAccuracy * 100.0))
    return charErrorRate

# TODO: Sort words in order of addition
def infer(model, fpath):
    """ Recognize text in image provided by file path
        Arguments:
            model - constructed model of NN,
	        fpath - path on file system of dir, where al images for infer are stored.
	"""
    # Get names of files in given dir to imgFiles
    if os.path.isdir(fpath):
        imgFiles = os.listdir(fpath)
    else:
        imgFiles = fpath  # If it a file - ??? (could it be?)
    recognized_words = []
    for (i, fnImg) in enumerate(imgFiles):
        print("File #", i, " Name: ", fnImg)
        print('Recognizing text from image %s...' % fnImg)
        # Check requirements for the image file to be processed by program
        if not check_file("%s/%s" % (fpath, fnImg)):
            continue
        img = preprocess(cv2.imread('%s%s' % (fpath, fnImg), cv2.IMREAD_GRAYSCALE), Model.imgSize)
        batch = Batch(None, [img])
        (recognized, probability) = model.inferBatch(batch, False)
        recognized_words.append(recognized[0])
        print('Recognized:', '"' + recognized[0] + '"')
        if probability:
            print('Probability:', probability[0])

    dump_results(recognized_words)


def dump_results(res):
    """ Dump(save) the output of the NN to txt file(s).
        Arguments:
            res - output of the NN, consist of strings
        Note: All files in "../dump/" is in .gitignore !!!
    """

    # If path do not exist create it
    if not os.path.isdir(ModelFilePaths.dumpDir):
        os.mkdir(ModelFilePaths.dumpDir)
    # If file do not exist create it and open in append mode
    str_to_write = str(' ').join(res) + "\n"
    with open(FilePaths.fnDumpRes, 'a+') as f:
        f.write(str_to_write)



if __name__ == '__main__':
	main()