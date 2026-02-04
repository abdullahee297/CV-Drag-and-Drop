# ---------------- Drag and Drop using Index Finger ----------------
# Index finger up  -> Drag
# Index + Middle   -> Drop

import cv2
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

# ---------------- Variables ----------------
ptime = 0
cw, ch = 1000, 600
color = (255, 0, 255)

# ---------------- Drag Rectangle Class ----------------
class DragRect:
    def __init__(self, poscenter, size=(100, 100)):
        self.poscenter = poscenter
        self.size = size
        self.dragging = False
        self.offset = (0, 0)

    def inside(self, cursor):
        box_x, box_y = self.poscenter
        w, h = self.size
        return (
            box_x - w // 2 < cursor[0] < box_x + w // 2 and
            box_y - h // 2 < cursor[1] < box_y + h // 2
        )

    def start_drag(self, cursor):
        box_x, box_y = self.poscenter
        self.offset = (box_x - cursor[0], box_y - cursor[1])
        self.dragging = True

    def stop_drag(self):
        self.dragging = False

    def update(self, cursor):
        if self.dragging:
            self.poscenter = (
                cursor[0] + self.offset[0],
                cursor[1] + self.offset[1]
            )

# ---------------- Hand Connections ----------------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)
cap.set(3, cw)
cap.set(4, ch)

rect = DragRect([150, 150])

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

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            img_h, img_w, _ = img.shape
            lm_list = []

            for lm in hand:
                lm_list.append((int(lm.x * img_w), int(lm.y * img_h)))

            # Draw hand
            for x, y in lm_list:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

            for s, e in HAND_CONNECTIONS:
                cv2.line(img, lm_list[s], lm_list[e], (0, 255, 0), 2)

            # -------- Finger detection --------
            finger = []

            # Thumb
            finger.append(1 if lm_list[4][0] > lm_list[3][0] else 0)

            # Other fingers
            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]
            for tip, pip in zip(tips, pips):
                finger.append(1 if lm_list[tip][1] < lm_list[pip][1] else 0)

            cursor = lm_list[8]

            # -------- Drag Logic --------
            # Grab
            if finger[1] and not finger[2]:
                if not rect.dragging and rect.inside(cursor):
                    rect.start_drag(cursor)
                rect.update(cursor)

            # Drop
            elif finger[1] and finger[2]:
                rect.stop_drag()

    # ---------------- Draw Rectangle ----------------
    box_x, box_y = rect.poscenter
    w, h = rect.size

    cv2.rectangle(
        img,
        (box_x - w // 2, box_y - h // 2),
        (box_x + w // 2, box_y + h // 2),
        color,
        cv2.FILLED
    )

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
