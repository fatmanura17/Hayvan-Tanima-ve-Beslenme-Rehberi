import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# AYARLAR
GUVEN_ESIGI = 70.0
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
DATASET_PATH = "dataset/animals"


# VERİ ARTTIRMA
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# KRİTİK: shuffle=False
val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

classes = list(train_generator.class_indices.keys())
num_classes = train_generator.num_classes

print("Sınıflar:", classes)
print("Sınıf sayısı:", num_classes)



# MODEL (TRANSFER LEARNING)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()



# MODEL EĞİTİMİ

print("\n--- Model Eğitimi Başladı ---")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)
print("--- Model Eğitimi Bitti ---")



# MODEL KAYDETME

os.makedirs("model", exist_ok=True)
model.save("model/hayvan_modeli.h5")


# GRAFİKLER
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Eğitim Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Eğitim Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.show()


plot_history(history)



# GELİŞMİŞ METRİKLER
print("\n--- Gelişmiş Metrikler ---")

val_generator.reset()

steps = val_generator.n // val_generator.batch_size
if val_generator.n % val_generator.batch_size != 0:
    steps += 1

Y_pred = model.predict(val_generator, steps=steps, verbose=1)
y_pred_classes = np.argmax(Y_pred, axis=1)

# DOĞRU ETİKETLER
y_true_classes = val_generator.classes


print("\nSınıflandırma Raporu")
print(classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=classes,
    zero_division=0
))

print("\nConfusion Matrix")
print(confusion_matrix(y_true_classes, y_pred_classes))


# TEK GÖRÜNTÜ TESTİ
print("\n--- Tek Görüntü Testi ---")

base_img_name = "yavruKedi"
possible_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]

img_path = None
for ext in possible_extensions:
    if os.path.exists(base_img_name + ext):
        img_path = base_img_name + ext
        break

if img_path is None:
    print("Test resmi bulunamadı, geçiyorum.")
else:
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)
    idx = np.argmax(pred)
    confidence = pred[0][idx] * 100

    print(f"Fotoğraf: {img_path}")
    print(f"Olasılık: %{confidence:.2f}")

# (Tahmin kısmı - Modelden gelen sonuçlar)
if confidence < GUVEN_ESIGI:
    print("Sonuç: Bilinmiyor (güven düşük)")
else:
    tahmin_edilen_hayvan = classes[idx] # Örneğin: 'kedi'
    print(f"Tahmin: {tahmin_edilen_hayvan}")
    print(f"Güven Oranı: %{confidence:.2f}")

# JSON BESLENME REHBERİ BAĞLANTISI 
    try:
       
        json_yolu = os.path.join("model", "animals_info.json") 
        
        with open(json_yolu, "r", encoding="utf-8") as f:
            rehber = json.load(f)
        
        # Tahmin edilen hayvanı rehberde aranıyor
        if tahmin_edilen_hayvan in rehber:
            print("\n" + "="*40)
            print(f" {tahmin_edilen_hayvan.upper()} BESLENME REHBERİ")
            print("-" * 40)
            print(rehber[tahmin_edilen_hayvan])
            print("="*40)
        else:
            print(f"\n {tahmin_edilen_hayvan} için rehberde beslenme bilgisi bulunamadı.")
            
    except FileNotFoundError:
        print(f"\n Hata: '{json_yolu}' dosyası bulunamadı! Lütfen dosyanın model klasöründe olduğundan emin olun.")


# TFLITE DÖNÜŞÜM
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model/hayvan_modeli.tflite", "wb") as f:
    f.write(tflite_model)

with open("model/labels.txt", "w") as f:
    for c in classes:
        f.write(c + "\n")

print("Model TFLite formatına çevrildi.")