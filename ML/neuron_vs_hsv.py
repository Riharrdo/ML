import numpy as np
import matplotlib.pyplot as plt
import cv2

img_bgr = cv2.imread("cheese.jpg")

if img_bgr is None:
    print("Не удалось загрузить изображение")
else:
    print(img_bgr.shape)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]

weights = np.array([2.0, -1.5, -1.5])
bias = -0.5
pixels = img_rgb.reshape(-1, 3).astype(np.float32) / 255.0 
z = pixels @ weights + bias
outputs = 1 / (1 + np.exp(-z))
neuron_mask = (outputs > 0.5).reshape(H, W).astype(np.uint8) * 255
neuron_result = cv2.bitwise_and(img_rgb, img_rgb, mask=neuron_mask)



img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
hsv_mask = cv2.bitwise_or(mask1, mask2)
hsv_result = cv2.bitwise_and(img_rgb, img_rgb, mask=hsv_mask)



diff_mask = cv2.bitwise_xor(hsv_mask, neuron_mask)
hsv_pixels = np.count_nonzero(hsv_mask)
neuron_pixels = np.count_nonzero(neuron_mask)
total_pixels = H * W

match_percent = (total_pixels - np.count_nonzero(diff_mask)) / total_pixels * 100

print(f"Количество пикселей HSV-маски: {hsv_pixels}")
print(f"Количество пикселей нейро-маски: {neuron_pixels}")
print(f"Процент совпадения масок: {match_percent:.2f}%")

plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Оригинал")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(hsv_mask, cmap='gray')
plt.title("Маска HSV")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(neuron_mask, cmap='gray')
plt.title("Маска нейрона")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(hsv_result)
plt.title("Результат HSV (красные области)")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(neuron_result)
plt.title("Результат нейрона")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(diff_mask, cmap='gray')
plt.title("Разница между масками (XOR)")
plt.axis('off')

plt.tight_layout()
plt.show()
