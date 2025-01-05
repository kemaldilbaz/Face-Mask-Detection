import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
import pickle

# GPU kullanılabilirliğini kontrol et
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU kullanıma hazır!")
    except RuntimeError as e:
        print(e)
else:
    print("GPU bulunamadı, CPU kullanılacak.")

# Mixed Precision Training (Karışık Hassasiyetli Eğitim)
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# YOLO için gerekli ayarlar
YOLO_INPUT_SIZE = 320  # Daha küçük giriş boyutu
NUM_CLASSES = 3
MAX_BOXES = 10  # Daha az bounding box sayısı
BATCH_SIZE = 16  # Batch size küçültüldü

# Verileri yükleme
train_images = np.load("C:/Users/KEMAL DİLBAZ/PycharmProjects/PythonProject/.venv1/train_images.npy")
test_images = np.load("C:/Users/KEMAL DİLBAZ/PycharmProjects/PythonProject/.venv1/test_images.npy")
with open("C:/Users/KEMAL DİLBAZ/PycharmProjects/PythonProject/.venv1/train_labels.pkl", "rb") as f:
    train_labels = pickle.load(f)
with open("C:/Users/KEMAL DİLBAZ/PycharmProjects/PythonProject/.venv1/test_labels.pkl", "rb") as f:
    test_labels = pickle.load(f)

# Veri ön işleme
def preprocess_yolo_image(image, target_size=YOLO_INPUT_SIZE):
    image = cv2.resize(image, (target_size, target_size))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def prepare_yolo_data(images, labels, num_classes=3):
    processed_images = np.array([preprocess_yolo_image(img)[0] for img in images])
    bbox_labels = []
    class_labels = []

    for label in labels:
        bbox = [b[1:] for b in label]
        classes = [0 if b[0] == "with_mask" else (1 if b[0] == "without_mask" else 2) for b in label]

        if len(bbox) > MAX_BOXES:
            bbox = bbox[:MAX_BOXES]
            classes = classes[:MAX_BOXES]
        else:
            padding_bbox = [[0, 0, 0, 0]] * (MAX_BOXES - len(bbox))
            padding_classes = [0] * (MAX_BOXES - len(classes))
            bbox.extend(padding_bbox)
            classes.extend(padding_classes)

        bbox_labels.append(bbox)
        class_labels.append(tf.keras.utils.to_categorical(classes, num_classes=num_classes))

    return processed_images, np.array(bbox_labels), np.array(class_labels)

# YOLO modeli oluşturma
def create_yolo_model(input_shape, num_classes):
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False  # EfficientNet katmanlarını dondur

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Bounding box çıktısı
    bbox_output = tf.keras.layers.Dense(MAX_BOXES * 4, activation="linear", name="bbox_flat")(x)
    bbox_output = tf.keras.layers.Reshape((MAX_BOXES, 4), name="bbox")(bbox_output)

    # Sınıf çıktısı
    class_output = tf.keras.layers.Dense(MAX_BOXES * num_classes, activation="linear", name="class_flat")(x)
    class_output = tf.keras.layers.Reshape((MAX_BOXES, num_classes), name="class")(class_output)

    return Model(inputs=base_model.input, outputs=[bbox_output, class_output])

# Modeli oluştur ve derle
yolo_model = create_yolo_model((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), NUM_CLASSES)
yolo_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={"bbox": "mse", "class": "categorical_crossentropy"},
    metrics={"bbox": "mae", "class": "accuracy"}
)

# Eğitim verilerini hazırla
train_images_yolo, train_bboxes_yolo, train_classes_yolo = prepare_yolo_data(train_images, train_labels)
test_images_yolo, test_bboxes_yolo, test_classes_yolo = prepare_yolo_data(test_images, test_labels)

# Model eğitimi
print("Model eğitimi başlıyor...")
history = yolo_model.fit(
    train_images_yolo,
    {"bbox": train_bboxes_yolo, "class": train_classes_yolo},
    validation_data=(test_images_yolo, {"bbox": test_bboxes_yolo, "class": test_classes_yolo}),
    batch_size=BATCH_SIZE,
    epochs=20,  # Daha az epoch ile test için
    verbose=1
)
print("Model eğitimi tamamlandı.")

# Modeli kaydet
yolo_model.save("yolo_face_mask_model.h5")
print("YOLO modeli başarıyla kaydedildi.")

# Model küçültme (Quantization)
print("Model küçültme işlemi...")
converter = tf.lite.TFLiteConverter.from_keras_model(yolo_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("yolo_face_mask_model.tflite", "wb") as f:
    f.write(tflite_model)
print("Model küçültme tamamlandı ve TFLite modeli kaydedildi.")
