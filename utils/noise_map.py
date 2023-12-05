import cv2


# img = cv2.imread('data/example.jpeg')

# blur = cv2.bilateralFilter(img,9,75,75)
# noise_map = img-blur


# img = cv2.imread(path)
def get_noise_map(path):
    img = cv2.imread(path)
    blur = cv2.blur(img, (5, 5)) #bilateralFilter(img,9,75,75)
    noise_map = img-blur
    return noise_map

# noise_map = get_noise_map(img)
# # cv2.imwrite('blur.jpg', blur)
# cv2.imwrite('noise_map.jpg', noise_map)