import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#calculate the angle between two lines
def calculate_angle(a,b,c):
    a = np.array(a) # hip
    b = np.array(b) # knee
    c = np.array(c) # ankle

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle
cap = cv2.VideoCapture(1)
#Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate angle
            angle = calculate_angle(hip, knee, ankle)

            # Visualize angle
            cv2.putText(image, str(angle), (50,150)
                           ,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            #visualize lines and points
            center_hip= tuple(np.multiply(hip, [640, 480]).astype(int))
            center_knee= tuple(np.multiply(knee, [640, 480]).astype(int))
            center_ankle= tuple(np.multiply(ankle, [640, 480]).astype(int))
            cv2.circle(image,center_hip,5, (255, 0, 0), -1)
            cv2.circle(image,center_knee,5, (255, 0, 0), -1)
            cv2.circle(image,center_ankle,5, (255, 0, 0), -1)

            cv2.line(image, center_hip, center_knee, (0,255, 0), 3)
            cv2.line(image, center_knee, center_ankle, (0,255, 0), 3)


            cv2.imshow('Sit to stand', image)

        except:
            pass



        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

    cap.release()
    cv2.destroyAllWindows()
