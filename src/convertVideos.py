import argparse
from util.poseModule import PoseDetector
from os import walk
from pathlib import Path
import os

def main(args):
    detector = PoseDetector()

    videos = next(walk(args.input), ([None, None, []]))[2]

    count = 0
    for video in videos:
        videoPath = Path(video)
        outputPath = videoPath.with_suffix('.txt')
        outputFile = os.path.join(args.output, outputPath)
        inputFile = os.path.join(args.input, video)

        if os.path.exists(outputFile):
            continue

        detector.videoToLandmarkFile(inputFile, outputFile)
        print(f'Created a data file for the video {video}')
        count += 1

    print(f'Finished converting videos. Converted {count} videos.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')

    main(parser.parse_args())
