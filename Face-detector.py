import cv2
from random import randrange

#load pretrainined data on face frontals from open cv and create a classifier
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces in
#img = cv2.imread('pic.jpeg')

webcam = cv2.VideoCapture(0)
key = cv2.waitKey(1)


#go through all frames in video;
while True:


    successful_frame_read, frame = webcam.read()

    #convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)



    # draw rectangles around the faces

    for (x, y, w, h) in face_coordinates:

        #will detect faces uses different colours.
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(150,225), randrange(150,225), randrange(150,225)), 2)

        #display image
    cv2.imshow('Eniolas face detector',frame)
    cv2.waitKey(1)

    #stop program when X is pressed

    if key==88 or key==120:
        break

webcam.release()


print ("Code Completed")
