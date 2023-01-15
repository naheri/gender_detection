from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
import cv2
import cvlib as cv
import argparse



# dimensions of images
img_width, img_height = (64, 64)
classes = ['man','woman']
# load model
model = load_model('models/model_0.968.h5')

def get_gender(frame):
    # load and resize image to 64x64
    frame = cv2.resize(frame, (img_width, img_height))
    # convert frame to numpy array
    frame = img_to_array(frame)
    # expand dimension of frame
    frame = np.expand_dims(frame, axis=0)
    print(model.predict(frame)[0][0])
    # making prediction with model
    return 'female' if model.predict(frame)[0][0] < 0.5 else 'male'


def gender_by_webcam():
    # Open the webcam
    webcam = cv2.VideoCapture(1)


    # Create a window called "webcam"
    cv2.namedWindow("webcam",cv2.WINDOW_NORMAL)
    # set the window size
    cv2.resizeWindow("webcam", 640, 480)

    # loop through frames
    while webcam.isOpened():

        # read frame from webcam 
        status, frame = webcam.read()
        
        # apply face detection
        face, confidence = cv.detect_face(frame)

        if len(face) <= 0:
            print('No face detected')
        else:
            print(f'{len(face)} faces detected')
        # loop through detected faces
        for f in face:
            print(face)
            print(list(enumerate(face)))
            # get corner points of face rectangle        
        # get corner points of face rectangle        
            # get corner points of face rectangle        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # draw rectangle over face
            cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

            # crop the detected face region
            face_crop = np.copy(frame[startY:endY,startX:endX])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue
            Y = startY - 10 if startY > 20 else startY + 10

            prediction = get_gender(face_crop)
            print(prediction)
            cv2.putText(frame, prediction, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        # display output
        cv2.imshow("webcam", frame)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release resources
    webcam.release()
    cv2.destroyAllWindows()

def gender_by_image(image_path):
    image = cv2.imread(image_path) # read image
     # Create a window called "webcam"
    cv2.namedWindow("gender detection by image",cv2.WINDOW_NORMAL)
    # set the window size
    cv2.resizeWindow("gender detection by image", 640, 480)
    
    # apply face detection
    face, confidence = cv.detect_face(image)

    if len(face) <= 0:
        print('No face detected')
    # loop through detected faces
    else:
        print(f'{len(face)} faces detected')
    for f in face:
        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(image[startY:endY,startX:endX])
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        
        Y = startY - 10 if startY > 20 else startY + 10
        prediction = get_gender(face_crop)
        print(prediction)
        cv2.putText(image, prediction, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # display output
    cv2.imshow("gender detection by image", image)
    print('image displayed')
    cv2.waitKey(0)

    # Release the window resources
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument('--image_path','--i', help="Path to the image to detect gender")
args = parser.parse_args()

image_path = args.image_path

if image_path is None:
    print('No image path provided')
    gender_by_webcam()
else:
    print('Image path provided')
    gender_by_image(image_path)



