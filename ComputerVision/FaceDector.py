from cv2 import cv2

# read image from local file
og_image = cv2.imread('./bla2.5577e4ec1f8e.jpg')

# transform image into grayscale for Viola-Jones algo
grayscale_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY)


# loading the classifieer and create a cascade object for face detection

face_cascade = cv2.CascadeClassifier(
    './haarcascade_frontalface_alt.xml')


detected_faces = face_cascade.detectMultiScale(grayscale_image)

# Drawing rectangles over the image based on pixel coordinate

'''

The original image
The coordinates of the top-left point of the detection
The coordinates of the bottom-right point of the detection
The color of the rectangle(a tuple that defines the amount of red, green, and blue(0-255))
The thickness of the rectangle lines

'''
for (column, row, width, height) in detected_faces:
    cv2.rectangle(
        og_image,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )


'''
imshow() displays the image. waitKey() waits for a keystroke. 
Otherwise, imshow() would display the image and immediately close the window. 
Passing 0 as the argument tells it to wait indefinitely. 
Finally, destroyAllWindows() closes the window when you press a key.

'''

# displaying image with detected faces
cv2.imshow('Image', og_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
