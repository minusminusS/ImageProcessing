import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os

def save_fig(name):
    plt.savefig(os.path.join("Data", name), dpi=300, bbox_inches='tight')


img_file = 'sar_1_gray.jpg'

# 1. Загрузите изображение в оттенках серого sar_3_gray.jpg.
img = io.imread(img_file)
#img_gray = io.imread(img_file, as_gray=True)
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

#print(img.shape) #(914, 787)
#img = img[1:1000, 400:700] # обрезать

# b,g,r = cv2.split(img)

plt.figure(figsize=(6,6))
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+0+0")

plt.suptitle('Гистограммы', fontsize=16)

# plt.subplot(2,3,4)
# plt.title('Blue')
# plt.imshow(b, cmap='gray')
# plt.subplot(2,3,5)
# plt.title('Green')
# plt.imshow(g, cmap='gray')
# plt.subplot(2,3,6)
# plt.title('Red')
# plt.imshow(r, cmap='gray')

# 2. постройте гистограмму

b_hist = cv2.calcHist([img_gray],[0],None,[256],(0,256), accumulate=False)
plt.subplot(2,3,1)
plt.title('Histogram')
plt.plot(b_hist)

b_hist_cum = b_hist.cumsum()
plt.subplot(2,3,2)
plt.title('Cumulative')
plt.plot(b_hist_cum)

b_hist_norm = b_hist /  (img.shape[0] * img.shape[1])
plt.subplot(2,3,3)
plt.title('Normalized')
plt.plot(b_hist_norm)

save_fig("histograms.png")


#plt.show()

# 3. реализуйте алгоритм гамма коррекции с параметром гамма <1, >1.

img_norm = img_gray.astype('float32')
img_norm = img_norm / img_norm.max()

def gamma_correction(img_, gamma):
    return np.power(img_, gamma)

gamma_low = 0.5
gamma_high = 2.0

img_gamma_low = gamma_correction(img_norm, gamma_low)
img_gamma_high = gamma_correction(img_norm, gamma_high)

img_gamma_low_8 = (img_gamma_low*255.).astype('uint8')
img_gamma_high_8 = (img_gamma_high*255.).astype('uint8')

plt.figure(figsize=(5,4))
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+800+0")

plt.suptitle('Гамма', fontsize=16)

plt.subplot(2,3,1)
plt.title('Gamma < 1')
plt.imshow(img_gamma_low_8, cmap='gray')
plt.subplot(2,3,2)
plt.title('Gamma > 1')
plt.imshow(img_gamma_high_8, cmap='gray')
plt.subplot(2,3,3)
plt.title('Orig')
plt.imshow(img_gray, cmap='gray')

save_fig("gamma.png")


#plt.show()

# 4. Сравните исходное изображение, скорректированное при помощи гамма-фильтра. MSE, SSIM.
from skimage.metrics import structural_similarity, mean_squared_error

img_gray_f = img_gray.astype('float32')
img_gray_f = img_gray_f / img_gray_f.max()
img_gamma_low_f = img_gamma_low_8.astype('float32') / 255.
img_gamma_high_f = img_gamma_high_8.astype('float32') / 255.


plt.figure(figsize=(5,4))
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+800+600")
plt.suptitle('Сравнение с скорректированными', fontsize=16)
# SSIM gamma < 1

(ssim_low, diff_low) = structural_similarity(img_gray_f, img_gamma_low_f, full=True, data_range=1.0)
diff_low = (diff_low * 255).astype("uint8")

print("Gamma < 1 (0.5)")
print("SSIM: {}".format(ssim_low))

plt.subplot(2,3,1)
plt.title('Gamma < 1')
plt.imshow(diff_low, cmap='gray')

# SSIM gamma < 1
(ssim_high, diff_high) = structural_similarity(img_gray_f, img_gamma_high_f, full=True, data_range=1.0)
diff_high = (diff_high * 255).astype("uint8")

print("Gamma > 1 (2)")
print("SSIM: {}".format(ssim_high))

plt.subplot(2,3,2)
plt.title('Gamma > 1')
plt.imshow(diff_high, cmap='gray')

save_fig("comparison.png")

#plt.show()

# MSE
mse_low = mean_squared_error(img_gray_f, img_gamma_low_f)
mse_high = mean_squared_error(img_gray_f, img_gamma_high_f)

print("MSE low: {}".format(mse_low))
print("MSE high: {}".format(mse_high))

# 5. реализуйте алгоритм статистической цветокоррекции на основе статистики eq_gray.

#eq_gray = io.imread('EQ.jpg', as_gray=True).astype('float32')
eq_gray = cv2.equalizeHist(img_gray.astype('uint8'))

src = img_gray.astype('float32')
src /= src.max()
eq = eq_gray.astype('float32')
eq /= eq.max()

mean_src = src.mean()
std_src = src.std()

mean_eq = eq.mean()
std_eq = eq.std()

img_corr = (std_eq / std_src) * (src - mean_src) + mean_eq
img_corr = np.clip(img_corr, 0, 1)
img_corr_8 = (img_corr * 255).astype('uint8')

plt.figure(figsize=(5,4))
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+1500+0")

plt.suptitle('Статическая цветокоррекция', fontsize=16)

plt.subplot(2,3,1)
plt.title('Corrected')
plt.imshow(img_corr_8, cmap='gray')
plt.subplot(2,3,2)
plt.title('Original')
plt.imshow(img_gray, cmap='gray')

save_fig("color_correction.png")



#plt.show()

# 6. Протестируйте работу алгоритмов пороговой фильтрации с различными параметрами.

thresholds = [50, 100, 150, 200]

img_u8 = (img_gray * 255).astype('uint8')

plt.figure(figsize=(6,6))
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+1500+600")
plt.suptitle('Пороговая фильтрация', fontsize=16)

for i, T in enumerate(thresholds):
    _, thresh1 = cv2.threshold(img_u8, T, 255, cv2.THRESH_BINARY)
    plt.subplot(2,3,i+1)
    plt.title('Threshold {}'.format(T))
    plt.imshow(thresh1, cmap='gray')

_, th_otsu = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.subplot(2,3,5)
plt.title('Otsu')
plt.imshow(th_otsu, cmap='gray')

th_adapt = cv2.adaptiveThreshold(img_u8, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,
                                 11, 2)
plt.subplot(2,3,6)
plt.title('Adaptive')
plt.imshow(th_adapt, cmap='gray')


save_fig("thresholds.png")
plt.show()
