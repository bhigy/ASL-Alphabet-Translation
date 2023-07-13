import cv2
import imutils
import mediapipe as mp
import numpy as np
import tensorflow as tf


CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]
model_asl = tf.keras.models.load_model("data/checkpoints/full.model")
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# Processing the input image
def process_image(img):
    # Converting the input to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(gray_image)

    # Returning the detected hands to calling function
    return results


# Drawing landmark connections
def draw_hand_connections(img, results):
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape

                # Finding the coordinates of each landmark
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Printing each landmark ID and coordinates
                # on the terminal
                print(id, cx, cy)

                # Creating a circle around each landmark
                cv2.circle(img, (cx, cy), 10, (0, 255, 0),
                           cv2.FILLED)
                # Drawing the landmark connections
                mpDraw.draw_landmarks(img, handLms,
                                      mpHands.HAND_CONNECTIONS)



def prepare(filepath):
    IMG_SIZE = 48
    new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def recognize(image, results):
    h, w, c = image.shape
    if results.multi_hand_landmarks:
        x = []
        y = []
        #for handLms in results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        for id, lm in enumerate(handLms.landmark):
            x.append(lm.x)
            y.append(lm.y)
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        margin_x = (x_max - x_min) / 2
        margin_y = (y_max - y_min) / 2
        c0_x = round(max(x_min - margin_x, 0) * w)
        c0_y = round(max(y_min - margin_y, 0) * h)
        c1_x = round(min(x_max + margin_x, 1) * w)
        c1_y = round(min(y_max + margin_y, 1) * h)

        cropped = image[c0_y:c1_y, c0_x:c1_x, :]
        frame1 = cv2.resize(cropped, (200, 200))
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        prediction = model_asl.predict([prepare(gray)])
        pred = (CATEGORIES[int(np.argmax(prediction[0]))])
        return pred, (c0_x, c0_y), (c1_x, c1_y)
    return None


def draw_prediction(image, pred):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, pred, (200,100), font, 1, (0,0,0), 2, cv2.LINE_AA)


def draw_rectangle(image, c0, c1):
    cv2.rectangle(image, c0, c1, (0,255,0),3)


def draw_crop(image, c0, c1):
    width = c1[0] - c0[0]
    height = c1[1] - c0[1]
    image[width:, height:, :] = image[c0[0]:c1[0], c0[1]:c1[1], :]


def main():
   # Replace 0 with the video path to use a
   # pre-recorded video
    cap = cv2.VideoCapture(0)

    while True:
        # Taking the input
        success, image = cap.read()
        image = imutils.resize(image, width=500, height=500)
        results = process_image(image)
        displayed_image = image.copy()
        draw_hand_connections(displayed_image, results)
        reco = recognize(image, results)

        if reco is not None:
            pred, c0, c1 = reco
            draw_prediction(displayed_image, pred)
            draw_rectangle(displayed_image, c0, c1)

        # Displaying the output
        cv2.imshow("Hand tracker", displayed_image)

        # Program terminates when q key is pressed
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
