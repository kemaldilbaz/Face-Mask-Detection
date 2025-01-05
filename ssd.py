import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pickle

# Maksimum bounding box sayısı
MAX_BOXES = 15
IMAGE_SIZE = 224

# Verileri yükleme
train_images = np.load("C:/Users/KEMAL DİLBAZ/PycharmProjects/PythonProject/.venv1/train_images.npy")
test_images = np.load("C:/Users/KEMAL DİLBAZ/PycharmProjects/PythonProject/.venv1/test_images.npy")
with open("C:/Users/KEMAL DİLBAZ/PycharmProjects/PythonProject/.venv1/train_labels.pkl", "rb") as f:
    train_labels = pickle.load(f)
with open("C:/Users/KEMAL DİLBAZ/PycharmProjects/PythonProject/.venv1/test_labels.pkl", "rb") as f:
    test_labels = pickle.load(f)


# Sınıf etiketlerini sabit boyut ve formatta düzeltme
def fix_class_labels(labels, num_classes=3):
    fixed_labels = []
    for label in labels:
        classes = [0 if b[0] == "with_mask" else (1 if b[0] == "without_mask" else 2) for b in label]
        if len(classes) > MAX_BOXES:
            classes = classes[:MAX_BOXES]
        else:
            padding_classes = [0] * (MAX_BOXES - len(classes))
            classes.extend(padding_classes)
        fixed_labels.append(tf.keras.utils.to_categorical(classes, num_classes=num_classes))
    return np.array(fixed_labels)


# Verileri SSD için ön işleme
def preprocess_for_ssd(images, labels, num_classes=3):
    images = images / 255.0
    bbox_labels = []
    class_labels = []

    for label in labels:
        bbox = [b[1:] for b in label]  # Sadece bounding box koordinatları
        if len(bbox) > MAX_BOXES:
            bbox = bbox[:MAX_BOXES]
        else:
            padding_bbox = [[0, 0, 0, 0]] * (MAX_BOXES - len(bbox))
            bbox.extend(padding_bbox)

        bbox_labels.append(bbox)

    class_labels = fix_class_labels(labels, num_classes)
    return images, (np.array(bbox_labels), class_labels)


train_images, (train_bboxes, train_classes) = preprocess_for_ssd(train_images, train_labels)
test_images, (test_bboxes, test_classes) = preprocess_for_ssd(test_images, test_labels)


# Veri jeneratörü
def data_generator(images, bboxes, classes, batch_size):
    while True:
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_bboxes = bboxes[i:i + batch_size]
            batch_classes = classes[i:i + batch_size]

            yield batch_images, {"bbox": batch_bboxes, "class": batch_classes}


# SSD Modeli
def create_ssd_model(input_shape, num_classes, max_boxes):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # Bounding box çıktısı
    bbox_output = layers.Dense(max_boxes * 4, activation="linear", name="bbox_flat")(x)
    bbox_output = layers.Lambda(lambda t: tf.reshape(t, [-1, max_boxes, 4]), name="bbox")(bbox_output)

    # Sınıf çıktısı
    class_output = layers.Dense(max_boxes * num_classes, activation="linear", name="class_flat")(x)
    class_output = layers.Lambda(lambda t: tf.reshape(t, [-1, max_boxes, num_classes]))(class_output)
    class_output = layers.Activation("softmax", name="class")(class_output)

    return Model(inputs=base_model.input, outputs=[bbox_output, class_output])


# Model oluşturma ve derleme
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
ssd_model = create_ssd_model(input_shape, num_classes=3, max_boxes=MAX_BOXES)
ssd_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={"bbox": "mse", "class": "categorical_crossentropy"},
    metrics={"bbox": "mae", "class": "accuracy"}
)

# Modeli eğitim
batch_size = 32
train_gen = data_generator(train_images, train_bboxes, train_classes, batch_size)
test_gen = data_generator(test_images, test_bboxes, test_classes, batch_size)

history = ssd_model.fit(
    train_gen,
    validation_data=test_gen,
    steps_per_epoch=len(train_images) // batch_size,
    validation_steps=len(test_images) // batch_size,
    epochs=80
)

# Modeli kaydetme
ssd_model.save("ssd_face_mask_model.h5")
print("Model başarıyla kaydedildi.")
