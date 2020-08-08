import cv2
import os
import numpy as np
import argparse


def get_images_path(path):
    images = os.listdir(path)
    for i, file in enumerate(images):
        if not file.endswith("jpg"):
            images.remove(file)
        images[i] = os.path.join(path, file)
    return images


DEFAULT_OUTPUT_PATH = os.path.join(os.getcwd(), "output")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", default=os.path.join(os.getcwd(), "images"),
                help="path to input image")
ap.add_argument("-s", "--sensitivity", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-a", "--ask", type=bool, default=False,
                help="ask about the face before blur")
ap.add_argument("-o", "--output", default=DEFAULT_OUTPUT_PATH,
                help="ask about the face before blur")
args = vars(ap.parse_args())


PROTOTXT = os.path.join(os.getcwd(), "deploy.prototxt.txt")
MODEL = os.path.join(os.getcwd(), "model.caffemodel")

IMAGES_PATH = args["images"]
CONFIDENCE = args["sensitivity"]
ASK = args["ask"]
OUTPUT_PATH = args["output"]
if not os.path.exists(OUTPUT_PATH):
    print(f"{OUTPUT_PATH} Invalid Output Path the output redirectd to {DEFAULT_OUTPUT_PATH}")   
    OUTPUT_PATH = DEFAULT_OUTPUT_PATH
    os.mkdir("output")

KSIZE = (23, 23)
SIGMAY = SIGMAX = 60


print("[INFO] loading images....")
images_path = get_images_path(IMAGES_PATH)
images = [cv2.imread(image) for image in images_path]

print("[INFO] loading model......")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

print("[INFO] start proccessing...")
all_faces = 0
for indx, image in enumerate(images, 1):
    h, w = image.shape[:2]

    print(f"[INFO] detecting image #{indx}")
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (320, 320)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    predictions = net.forward()

    blured = faces = 0
    for i in range(predictions.shape[2]):
        confidence = predictions[0, 0, i, 2]

        if confidence < CONFIDENCE:
            continue

        box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        sub_face = image[startY:endY, startX:endX]

        if len(sub_face):
            faces += 1
            key = 13
            print(f"[PHOTO] face detected {faces}")
            if ASK:
                cv2.imshow("face check", sub_face)
                key = cv2.waitKey(0) & 0xff
            if key == 13:
                blured += 1
                sub_face = cv2.GaussianBlur(sub_face, KSIZE, SIGMAX, SIGMAY)
                image[startY:startY+sub_face.shape[0],
                      startX:startX+sub_face.shape[1]] = sub_face
        cv2.destroyAllWindows()
    all_faces += faces
    print(f"[PHOTO] {faces} Face Detected")
    print(f"[PHOTO] {blured} Face Blured")
    cv2.imwrite(os.path.join(OUTPUT_PATH, f"{indx}.jpg"), image)


print(f"[INFO] {all_faces} Face Detected")
