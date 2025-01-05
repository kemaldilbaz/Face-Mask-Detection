import cv2
import os
import numpy as np

# Giriş ve çıkış dizinleri
input_dir = r"D:\images"  # Görsellerin olduğu dizin
output_dir = r"D:\processed_images"  # İşlenmiş görsellerin kaydedileceği dizin
os.makedirs(output_dir, exist_ok=True)

# Sabit boyut
target_size = (300, 300)  # SSD için
# target_size = (416, 416)  # YOLO için

# Görselleri işleme
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Görseli yükle
        image = cv2.imread(img_path)

        # Yeniden boyutlandırma
        resized_image = cv2.resize(image, target_size)

        # Normalizasyon (0-1 aralığına)
        normalized_image = resized_image / 255.0

        # Kaydet (isteğe bağlı, normalizasyon sonrası float32 veriler genelde kaydedilmez)
        cv2.imwrite(output_path, (normalized_image * 255).astype(np.uint8))

        print(f"Processed {filename} and saved to {output_path}")
