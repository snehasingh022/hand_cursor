import cv2  # type: ignore
import mediapipe  # type: ignore
import pyautogui  # type: ignore

capture_hands = mediapipe.solutions.hands.Hands()
drawing_option = mediapipe.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
camera = cv2.VideoCapture(0)

# Define active region of your hand (calibrate if needed)
HAND_X_MIN = 100
HAND_X_MAX = 500
HAND_Y_MIN = 100
HAND_Y_MAX = 400

x1 = y1 = x2 = y2 = x3 = y3 = 0

while True:
    _, image = camera.read()
    image = cv2.flip(image, 1)
    image_height, image_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_hands = capture_hands.process(rgb_image)
    all_hands = output_hands.multi_hand_landmarks

    if all_hands:
        for hand in all_hands:
            drawing_option.draw_landmarks(image, hand)
            one_hand_landmark = hand.landmark

            for id, lm in enumerate(one_hand_landmark):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)

                if id == 8:  # Index finger tip
                    # Clamp to active area
                    x_clamped = max(HAND_X_MIN, min(x, HAND_X_MAX))
                    y_clamped = max(HAND_Y_MIN, min(y, HAND_Y_MAX))

                    # Stretch active hand region to screen
                    mouse_x = screen_width * (x_clamped - HAND_X_MIN) / (HAND_X_MAX - HAND_X_MIN)
                    mouse_y = screen_height * (y_clamped - HAND_Y_MIN) / (HAND_Y_MAX - HAND_Y_MIN)

                    cv2.circle(image, (x, y), 10, (0, 255, 255), -1)
                    pyautogui.moveTo(mouse_x, mouse_y, duration=0.01)
                    x1, y1 = x, y

                if id == 4:  # Thumb tip
                    x2, y2 = x, y
                    cv2.circle(image, (x, y), 10, (255, 0, 0), -1)

                if id == 12:  # Middle finger tip for scrolling
                    x3, y3 = x, y
                    cv2.circle(image, (x, y), 10, (0, 0, 255), -1)

        # Clicking condition
        dist_click = y2 - y1
        if dist_click < 20:
            pyautogui.click()
            pyautogui.sleep(0.2)

        # Scrolling condition
        dist_scroll = y3 - y1
        if abs(dist_scroll) > 40:
            if dist_scroll > 0:
                pyautogui.scroll(-50)  # Scroll down
            else:
                pyautogui.scroll(50)   # Scroll up
            pyautogui.sleep(0.1)

    cv2.imshow("Hand movement video capture", image)
    key = cv2.waitKey(100)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
