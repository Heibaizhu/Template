import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os

def FFT(img):
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    return magnitude

dehazed_dir = '/mnt/data/zhajunwei/JDRNN2/experiments/MFAM/visualization'
# dehazed_dir = '/mnt/data/zhajunwei/JDRNN2/experiments/MFAM-pure/visualization'
with open('/mnt/data/zhajunwei/JDRNN/metadata/RESIDE-STANDARD-ITS-TEST.json', 'r') as f:
    data = json.load(f)
hazes = data['haze']
clears = data['clear']

fft_error = 0
N = len(hazes)
for haze, clear in zip(hazes, clears):
    name = os.path.basename(haze)
    dehazed = os.path.join(dehazed_dir, name)
    data_clear = cv2.imread(clear)
    data_dehazed = cv2.imread(dehazed)
    error = cv2.absdiff(data_dehazed, data_clear) / 255
    # error = np.abs(data_dehazed - data_clear) / 255
    fft_error += FFT(error)

# for clear in clears:
#     data_clear = cv2.imread(clear)
#     name = os.path.basename(clear)
#     '_'.join(name.split('_')[:2]) + '.png'
#     dehazed = os.path.join(dehazed_dir, name)
#     data_dehazed = cv2.imread(dehazed)
#     error = np.abs(data_dehazed - data_clear) / 255
#     fft_error += FFT(error)

log_fft_error = np.log(fft_error / N)
log_fft_error = np.uint8(log_fft_error / 9.97 * 255)
log_fft_error = cv2.applyColorMap(log_fft_error, colormap=cv2.COLORMAP_JET)
log_fft_error = cv2.cvtColor(log_fft_error, cv2.COLOR_BGR2RGB)

plt.imshow(log_fft_error)
plt.title("Magnitude Spectrum")
plt.savefig('Fre-embedding')
#
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(mag, cmap='gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()
