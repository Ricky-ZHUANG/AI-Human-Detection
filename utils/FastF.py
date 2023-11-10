import cv2
from matplotlib import pyplot as plt
import numpy as np

# img = cv2.imread('data/example.jpeg',0)
# dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

# cv2.imwrite('Magnitude Spectrum.jpg', magnitude_spectrum)

def get_magnitude_spectrum(path):
    img = cv2.imread(path,0)
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    return magnitude_spectrum

# magnitude_spectrum = get_magnitude_spectrum(img)
# cv2.imwrite('Magnitude Spectrum.jpg', magnitude_spectrum)
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()