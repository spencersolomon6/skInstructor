import os.path

import numpy as np

from util.poseModule import PoseDetector
import cv2
import time
import argparse
import util.RNN
import json
from os import walk


def main(args):
    verbose = args.verbose or False
    train = args.train or False
    demo = args.demo or False

    rnn = util.RNN.RNN(verbose=verbose)
    detector = PoseDetector()

    if demo:
        detector.displayVideo('data/training_videos/vid1.mp4')

    if train:
        trainingDir = 'data/training_data'
        dataFiles = next(walk(trainingDir), ([None, None, []]))[2]

        for filename in dataFiles:
            batchedData = np.array([])
            with open(os.path.join(trainingDir, filename)) as f:
                data = json.load(f)
                print(data.shape)
                batchedData = np.append(batchedData, data)
                print(batchedData)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='skInstructor',
        description='An AI driven ski instructor',
        epilog='Go to the Github for more.'
    )

    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--demo', action='store_true')

    main(parser.parse_args())