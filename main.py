#----------------------Drag and Drop Using Index Finger---------------

# ========Logic==========
# Index finger move it
# 2 fingers stop it at that position


import cv2
import cvzone
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------------- MediaPipe Setup ----------------
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

#-----------Variables--------------

ptime = 0
cw = 1000
ch = 600
color = (255, 0, 255)
box_x, box_y, wx, hx = 50, 50, 100, 100


class DragRect():
    def __init__ (self, poscenter, size=[100,100]):
        self.poscenter = poscenter
        self.size = size
        # self.default_color = (255, 0, 255)
        # self.hover_color = (0, 255, 0)
        # self.color = self.default_color
        
    def update(self, cursor):
        box_x, box_y = self.poscenter
        wx, hx = self.size
        if (box_x - wx//2 < cursor[0] < box_x + wx//2 and
            box_y - hx//2 < cursor[1] < box_y + hx//2):
            self.poscenter = cursor
            
    # def hover(self, cursor):
    #     if self.update(cursor):
    #         self.color = self.hover_color
    #     else:
    #         self.color = self.default_color



# ---------------- Hand Connections ----------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]


cap = cv2.VideoCapture(0)
cap.set(3, cw)
cap.set(4, ch)

rectList = []
for x in range(5):
    rectList.append(DragRect([x*150 + 150, 150]))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if not success:
        break
    
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)

    result = detector.detect(mp_image)

    finger_count = 0
    finger = []

    if result.hand_landmarks:
        for hand in result.hand_landmarks:

            h, w, _ = img.shape
            lm_list = []
            x_list = []
            y_list = []


            for lm in hand:
                lm_list.append((int(lm.x * w), int(lm.y * h)))
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            for x, y in lm_list:
                cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
                
            if len(lm_list) >= 21:
                x1, y1 = lm_list[8]

            for start, end in HAND_CONNECTIONS:
                cv2.line(img, lm_list[start], lm_list[end], (0, 255, 0), 2)
                
            
            if lm_list[4][0] > lm_list[3][0]:
                finger.append(1)
                finger_count += 1
            else:
                finger.append(0)
                
                
            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]
            
            for tip, pip in zip(tips, pips):
                if lm_list[tip][1] < lm_list[pip][1]:
                    finger.append(1)
                    finger_count += 1
                else: 
                    finger.append(0)
            # print(finger_count, finger)
            cursor = lm_list[8]                
            
            if finger[1] and finger [2]:
                print("Drop")
                
            if finger[1] and not finger[2]:
                for rect in rectList:
                    rect.update(cursor)
                                        
                    # print("Drag")

    for rect in rectList:
        box_x, box_y = rect.poscenter
        wx, hx = rect.size
        cv2.rectangle(
            img, 
            (box_x - wx//2, box_y - hx//2),
            (box_x + wx//2, box_y + hx//2),
            color, 
            cv2.FILLED
        )
        cvzone.cornerRect(
            img, 
            (box_x - wx//2, box_y - hx//2, wx, hx),
            20, 
            rt = 0
        )
            
    #---------------Text on window----------------

    # FPS
    cv2.putText(
        img,
        f'FPS: {int(fps)}',
        (10, 30),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Drag and Drop", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
