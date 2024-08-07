
import cv2
img = cv2.imread(r'D:\MLP\StegaINR-main\data\celeba_hq\train\000000001781.jpg')


height, width, _ = img.shape

dst_shape = 256
r = min(height // dst_shape, width // dst_shape)

normal_height = r * dst_shape
normal_width = r * dst_shape

img = img[(height - normal_height) // 2 : (height - normal_height) // 2 + normal_height,
         (width - normal_width) // 2 : (width - normal_width) // 2 + normal_width]

img = cv2.resize(img, (dst_shape, dst_shape))
cv2.imwrite(r'D:\MLP\StegaINR-main\data\celeba_hq\train\secret154.jpg', img)