import mediapipe as mp
import cv2
import time
import numpy as np
import json

# TODO : Add averaging for hand and foot positions
LANDMARK_DICT = {
    0: "nose",
    11: "shoulder_l",
    12: "shoulder_r",
    13: "elbow_l",
    14: "elbow_r",
    15: "wrist_l",
    16: "wrist_r",
    23: "hip_l",
    24: "hip_r",
    25: "knee_l",
    26: "knee_r",
    27: "ankle_l",
    28: "ankle_r"
}

class PoseDetector:

    def __init__(self, mode = False, upBody = False, smooth=True, detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self, img, draw=True):
        landmarks = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id in LANDMARK_DICT.keys():
                    landmarks.append([cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        return landmarks

    def videoToLandmarkFile(self, videoFile, landmarkFile):
        landmarks = []
        try:
            video = cv2.VideoCapture(videoFile)

            success = True
            while success:
                success, screenshot = video.read()
                self.findPose(screenshot, False)
                lm = self.getPosition(screenshot, False)
                landmarks.append(lm)
        except:
            pass

        with open(landmarkFile, 'w') as output:
            json.dump(landmarks, output)
