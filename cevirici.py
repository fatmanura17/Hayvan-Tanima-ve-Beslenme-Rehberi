import tensorflow as tf
import os

h5_path = 'model/hayvan_modeli.h5'
tflite_path = 'model/hayvan_modeli.tflite'
dataset_path = 'dataset/animals' # klaasör isimlerini aldığımız yer

def convert_model():
    # 1. klasör isimlerinden labelsları bul
    if os.path.exists(dataset_path):
        # klasörleri bul ve alfabetik sırala 
        labels = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        
        # labels.txt dosyasını kaydet
        with open('model/labels.txt', 'w') as f:
            for label in labels:
                f.write(label + '\n')
        print(f"Labels dosyası oluşturuldu: {labels}")
    else:
        print("UYARI: Dataset klasörü bulunamadı, labels.txt oluşturulamadı!")

    # 2. kaydedilmiş .h5 modelini yükle
    print("Model yükleniyor...")
    if not os.path.exists(h5_path):
        print("HATA: .h5 dosyası bulunamadı! Önce eğitimi tamamlayıp kaydetmelisin.")
        return

    model = tf.keras.models.load_model(h5_path)

    # 3. TFLite'a Dönüştür
    print("Dönüştürme işlemi başlıyor...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # --- MAC ÇÖKMESİNİ ENGELLEYEN AYARLAR ---
    # Bu ayarlar modelin boyutunu düşürür ve uyumluluğu artırır
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    try:
        tflite_model = converter.convert()
        
        # 4. Dosyayı Kaydet
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"\nBAŞARILI! Model şuraya kaydedildi: {tflite_path}")
        print("Artık mobil uygulamaya geçebilirsin.")
        
    except Exception as e:
        print(f"\nDönüştürme hatası: {e}")
        print("Alternatif yöntem deneniyor...")
        # Eğer yukarıdaki çalışmazsa, eski dönüştürücüyü dene (Yedek plan)
        try:
            converter.experimental_new_converter = False
            tflite_model = converter.convert()
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            print("Yedek yöntemle başarıldı!")
        except Exception as e2:
            print(f"Yedek yöntem de başarısız oldu: {e2}")

if __name__ == "__main__":
    convert_model()