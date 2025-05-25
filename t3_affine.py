import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def find_dest_points(p1, p2, p3, p4):
    widthA = np.linalg.norm(p1 - p2)
    widthB = np.linalg.norm(p3 - p4)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(p1 - p3)
    heightB = np.linalg.norm(p2 - p4)
    maxHeight = max(int(heightA), int(heightB))

    return np.array([[0, 0], [maxWidth, 0], [0, maxHeight], [maxWidth, maxHeight]], dtype=np.float32)

def transform_perspective(img, src_points, dst_points):
    M = cv.getPerspectiveTransform(src_points, dst_points)
    width = int(dst_points[1][0])
    height = int(dst_points[2][1])
    return cv.warpPerspective(img, M, (width, height))

def read_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def main(image_path):
    img = read_image(image_path)

    # Display the image and let user click 4 points
    plt.imshow(img)
    plt.title("Click 4 points: Top-left, Top-right, Bottom-left, Bottom-right")
    points = plt.ginput(4, timeout=0)
    plt.close()

    if len(points) != 4:
        print("You must select exactly 4 points.")
        return

    pts1 = np.array(points, dtype=np.float32)
    pts2 = find_dest_points(*pts1)

    warped = transform_perspective(img, pts1, pts2)

    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.subplot(122), plt.imshow(warped), plt.title('Warped')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perspective transform using matplotlib clicks.")
    parser.add_argument("image_path", help="Path to the image.")
    args = parser.parse_args()
    main(args.image_path)
