import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os

def save_fig(name):
    plt.savefig(os.path.join("../Data", name), dpi=300, bbox_inches='tight')


# Загружаем изображение
name = 'img.jpg'
image = cv2.imread(name)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 1) Зашумить изображение при помощи шума гаусса, постоянного шума.

plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.title("Исходное")
plt.imshow(image_gray, cmap='gray')


# Шум Гаусса
mean = 0
stddev = 100
gauss_noise = np.zeros_like(image_gray, dtype=np.uint8)
cv2.randn(gauss_noise, mean, stddev)

image_gauss_noisy = cv2.add(image_gray, gauss_noise)

plt.subplot(1,3,2)
plt.title("Гауссовский шум")
plt.imshow(image_gauss_noisy, cmap='gray')


# Постоянный шум
constant_noise = np.random.randint(0, 50, size=image_gray.shape, dtype=np.uint8)
image_const_noisy = cv2.add(image_gray, constant_noise)

plt.subplot(1,3,3)
plt.title("Постоянный шум")
plt.imshow(image_const_noisy, cmap='gray')

save_fig("First.png")

# 2) Протестировать медианный фильтр, фильтр гаусса, билатериальный фильтр, фильтр нелокальных средних с различными параметрами.

# Медианный фильтр
median_gauss = cv2.medianBlur(image_gauss_noisy, 5)
median_const = cv2.medianBlur(image_const_noisy, 5)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Медианный (гауссовский шум)")
plt.imshow(median_gauss, cmap='gray')


plt.subplot(1,2,2)
plt.title("Медианный (постоянный шум)")
plt.imshow(median_const, cmap='gray')

save_fig("Mediam.png")

# Гауссовский фильтр
gauss_f_gauss = cv2.GaussianBlur(image_gauss_noisy, (5,5), 1.5)
gauss_f_const = cv2.GaussianBlur(image_const_noisy, (5,5), 1.5)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Гауссовский фильтр (гауссовский шум)")
plt.imshow(gauss_f_gauss, cmap='gray')


plt.subplot(1,2,2)
plt.title("Гауссовский фильтр (постоянный шум)")
plt.imshow(gauss_f_const, cmap='gray')

save_fig("Gaus.png")

# Билатеральный фильтр
bilateral_gauss = cv2.bilateralFilter(image_gauss_noisy, d=9, sigmaColor=75, sigmaSpace=75)
bilateral_const = cv2.bilateralFilter(image_const_noisy, d=9, sigmaColor=75, sigmaSpace=75)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Билатеральный (гауссовский шум)")
plt.imshow(bilateral_gauss, cmap='gray')


plt.subplot(1,2,2)
plt.title("Билатеральный (постоянный шум)")
plt.imshow(bilateral_const, cmap='gray')

save_fig("Bilater.png")


# Нелокальные средние
nlm_gauss = cv2.fastNlMeansDenoising(image_gauss_noisy, None, h=15, templateWindowSize=7, searchWindowSize=21)
nlm_const = cv2.fastNlMeansDenoising(image_const_noisy, None, h=15, templateWindowSize=7, searchWindowSize=21)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("NLM (гауссовский шум)")
plt.imshow(nlm_gauss, cmap='gray')


plt.subplot(1,2,2)
plt.title("NLM (постоянный шум)")
plt.imshow(nlm_const, cmap='gray')

save_fig("NoLocal.png")

# Выяснить, какой фильтр показал лучший результат фильтрации шума.
def evaluate_filter(original, filtered):
    psnr = peak_signal_noise_ratio(original, filtered)
    ssim = structural_similarity(original, filtered)
    return psnr, ssim

# Оценка для гауссовского шума
print("Гауссовский шум:")
print("Median:", evaluate_filter(image_gray, median_gauss))
print("Gaussian:", evaluate_filter(image_gray, gauss_f_gauss))
print("Bilateral:", evaluate_filter(image_gray, bilateral_gauss))
print("NLM:", evaluate_filter(image_gray, nlm_gauss))

# Оценка для постоянного шума
print("\nПостоянный шум:")
print("Median:", evaluate_filter(image_gray, median_const))
print("Gaussian:", evaluate_filter(image_gray, gauss_f_const))
print("Bilateral:", evaluate_filter(image_gray, bilateral_const))
print("NLM:", evaluate_filter(image_gray, nlm_const))


plt.show()
