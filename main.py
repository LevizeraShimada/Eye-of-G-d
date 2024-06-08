import cv2
from simple_facerec import SimpleFacerec
import pyautogui
import time
from datetime import datetime, timedelta
import os
import pygetwindow as gw


def find_camera_window():
    windows = gw.getWindowsWithTitle("Face Recognition - Basis")
    if windows:
        return windows[0]
    else:
        print("Janela da câmera não encontrada.")
        return None


cap = cv2.VideoCapture(0)

facerec = SimpleFacerec()

facerec.load_encoding_images("data_img/")

while True:
    ret, frame = cap.read()

    face_locations, face_names = facerec.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        print(face_loc)
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        if x1 <= 75:
            dt_string = datetime.now().strftime("%Y%m%d %H%M%S")
            save_dir = r"C:/Users/levis/Documents/prints/"
            last_screenshot_time = datetime.min  # Inicializa com a menor data possível
            screenshot_interval = timedelta(seconds=10)  # Intervalo de 10 segundos entre capturas

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f"{name}_{dt_string}.jpeg")
            if datetime.now() - last_screenshot_time >= screenshot_interval:
                cw = find_camera_window()
                if cw:
                    left, top, width, height = cw.left, cw.top, cw.width, cw.height
                    screenshot = pyautogui.screenshot(region=(left, top, width, height))
                    screenshot.save(save_path, "jpeg")
                    last_screenshot_time = datetime.now()

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Face Recognition - Basis", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
