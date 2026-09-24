import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread("photo.jpg")

if img_bgr is None:
    print("Не удалось загрузить изображение")
else:
    print(img_bgr.shape)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(5, 5))
plt.imshow(img_rgb)
plt.title('Оригинал')
plt.axis('off')
plt.show()

def sepia_manual(img):
    result = np.zeros_like(img, dtype=np.float32)
    h, w = img.shape[:2]
    for y in range(h):
        for x in range(w):
            r = float(img[y, x, 0])
            g = float(img[y, x, 1])
            b = float(img[y, x, 2])
            
            r_new = 0.393 * r + 0.769 * g + 0.189 * b
            g_new = 0.349 * r + 0.686 * g + 0.168 * b
            b_new = 0.272 * r + 0.534 * g + 0.131 * b
            
            result[y, x, 0] = min(255, r_new)
            result[y, x, 1] = min(255, g_new)
            result[y, x, 2] = min(255, b_new)
    return result.astype(np.uint8)

img_manual = sepia_manual(img_rgb)

sepia_matrix = np.array([[0.393, 0.769, 0.189],
                         [0.349, 0.686, 0.168],
                         [0.272, 0.534, 0.131]], dtype=np.float32)

img_lib = cv2.transform(img_rgb, sepia_matrix)

plt.figure(figsize=(13, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Оригинал')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_manual)
plt.title('Не оригинал')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_lib)
plt.title('Не оригинал, но по ГОСТу(почти)')
plt.axis('off')

plt.show()
