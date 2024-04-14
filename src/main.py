from util.poseModule import PoseDetector
import cv2
import time
import argparse
import util.RNN


def main(args):
    verbose = args.verbose or False
    train = args.train or False
    demo = args.demo or False

    cap = cv2.VideoCapture('data/vid1.mp4')  # make VideoCapture(0) for webcam
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 1200, 1200)

    rnn = util.RNN.RNN(verbose=verbose)

    pTime = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img, False)
        landmarks = detector.getPosition(img, False)

        prediction = rnn.predict(landmarks)

        print(prediction)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        if False:
            cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow("Image", img)
            cv2.waitKey(10000)


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