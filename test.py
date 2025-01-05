import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

# Custom layer definitions
class Cast(Layer):
    def call(self, inputs, **kwargs):
        return tf.cast(inputs, tf.float32)

# Modellerin yolları
YOLO_MODEL_PATH = "yolo_face_mask_model.h5"
SSD_MODEL_PATH = "ssd_face_mask_model.h5"

# Test görüntülerinin bulunduğu klasör
TEST_IMAGE_FOLDER = "D:/images"

# Sınıf etiketleri
labels = ["with_mask", "without_mask", "mask_weared_incorrect"]

# YOLO ve SSD modellerini yükleme

def load_yolo_model():
    print("YOLO modeli yükleniyor...")
    yolo_model = load_model(YOLO_MODEL_PATH, custom_objects={
        'mse': tf.keras.losses.MeanSquaredError(),
        'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy(),
        'Cast': Cast
    })
    print("YOLO modeli başarıyla yüklendi.")
    return yolo_model

def load_ssd_model():
    print("SSD modeli yükleniyor...")
    ssd_model = load_model(SSD_MODEL_PATH, custom_objects={
        'mse': tf.keras.losses.MeanSquaredError(),
        'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy(),
        'Cast': Cast
    })
    print("SSD modeli başarıyla yüklendi.")
    return ssd_model

# Görselleri test etmek için yardımcı fonksiyonlar

def preprocess_image(image, target_size):
    """Görüntüyü yeniden boyutlandır ve normalizasyon uygula."""
    image = cv2.resize(image, (target_size, target_size))
    return np.expand_dims(image / 255.0, axis=0)

def scale_bounding_boxes(bboxes, image_shape, input_size):
    """
    Bounding box koordinatlarını giriş boyutundan gerçek görüntü boyutlarına ölçeklendirir.
    """
    h, w, _ = image_shape
    scale_x = w / input_size
    scale_y = h / input_size

    scaled_bboxes = []
    for bbox in bboxes:
        x_min = int(bbox[0] * scale_x)
        y_min = int(bbox[1] * scale_y)
        x_max = int(bbox[2] * scale_x)
        y_max = int(bbox[3] * scale_y)
        scaled_bboxes.append([x_min, y_min, x_max, y_max])

    return scaled_bboxes

def draw_bounding_boxes(image, bboxes, classes, confidences, labels):
    """
    Görüntüye bounding box çizer ve sınıf etiketlerini ekler.
    """
    for bbox, cls, conf in zip(bboxes, classes, confidences):
        x_min, y_min, x_max, y_max = bbox
        label = f"{labels[cls]}: {conf:.2f}"
        # Bounding box çiz
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Etiketi ekle
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Test görüntülerini işleme

def test_model_on_images(model, input_size, labels, test_folder):
    threshold = 0.5  # Confidence threshold
    for filename in os.listdir(test_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(test_folder, filename)
        image = cv2.imread(image_path)
        input_image = preprocess_image(image, input_size)
        predictions = model.predict(input_image)

        try:
            bbox_predictions, class_predictions = predictions
        except ValueError as e:
            print(f"Tahmin formatı hatası: {e}")
            continue

        bboxes = scale_bounding_boxes(bbox_predictions[0], image.shape, input_size)
        class_probs = class_predictions[0]
        classes = np.argmax(class_probs, axis=-1)
        confidences = np.max(class_probs, axis=-1)

        filtered_bboxes, filtered_classes, filtered_confidences = [], [], []
        for bbox, cls, conf in zip(bboxes, classes, confidences):
            if conf >= threshold:
                filtered_bboxes.append(bbox)
                filtered_classes.append(cls)
                filtered_confidences.append(conf)

        output_image = draw_bounding_boxes(image, filtered_bboxes, filtered_classes, filtered_confidences, labels)
        cv2.imshow("Tahmin Sonuçları", output_image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



# Ana akış
if __name__ == "__main__":
    # YOLO modeli yükle ve test et
    yolo_model = load_yolo_model()
    print("YOLO modeli test ediliyor...")
    test_model_on_images(yolo_model, 320, labels, TEST_IMAGE_FOLDER)

    # SSD modeli yükle ve test et
    ssd_model = load_ssd_model()
    print("SSD modeli test ediliyor...")
    test_model_on_images(ssd_model, 224, labels, TEST_IMAGE_FOLDER)
