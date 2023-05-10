import cv2
import numpy as np
import os

def is_yellow(car_frame):
    # BRG -> HSV color space
    hsv = cv2.cvtColor(car_frame, cv2.COLOR_BGR2HSV)

    # yellow in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # check for yellow mask
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return np.any(mask)

def main():
    cap = cv2.VideoCapture(0) # live video => 0
    # car cascade classifier XML
    car_cascade = cv2.CascadeClassifier('cars.xml')

    while True:
        ret, frames = cap.read()
        if not ret:
            break

        # car detection
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 9)

        for (x, y, w, h) in cars:
            car_frame = frames[y:y + h, x:x + w]

            if is_yellow(car_frame):
                print("YELLOW CAR")
                os.system('say "YELLOW CAR"')

                # draw high light box
                # cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.rectangle(frames, (x, y - 40), (x + w, y), (0, 255, 0), -2)
                # cv2.putText(frames, 'Yellow Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # cv2.imshow('car', car_frame)

        frames = cv2.resize(frames, (600, 400))
        cv2.imshow('Car Detection System', frames)

        # exit with "esc"
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
