import cv2


def show_detected_face(image_path, face_locations):
    image = cv2.imread(image_path)
    for (x, y, w, h) in face_locations:
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('opencv_image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
