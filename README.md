# Akıllı Hayvan Tanıma ve Beslenme Rehberi

Bu proje, **Bilecik Şeyh Edebali Üniversitesi Bilgisayar Mühendisliği** bitirme çalışması kapsamında geliştirilmiştir. Derin öğrenme teknikleri kullanılarak 84 farklı hayvan türünü tanıyabilen ve bu türlere yönelik beslenme önerileri sunan bir sistem altyapısıdır.

## 🚀 Proje Hakkında
Sistem, görüntü sınıflandırma (Image Classification) problemini çözmek için **MobileNetV2** mimarisini kullanır. Model, mobil cihazlarda verimli çalışabilmesi için **TensorFlow Lite (TFLite)** formatına optimize edilmiştir.

## 🛠 Kullanılan Teknolojiler
* **Programlama Dili:** Python
* **Derin Öğrenme Framework:** TensorFlow & Keras
* **Mimari:** MobileNetV2 (Transfer Learning)
* **Veri İşleme:** OpenCV, NumPy, Pandas
* **Model Formatı:** .h5, .tflite, JSON (Bilgi bankası)

## 📊 Performans
* **Doğruluk Oranı (Accuracy):** %90.2
* **Güven Eşiği (Confidence Threshold):** %70

## 📂 Dosya Yapısı
* `model/`: Eğitilmiş model dosyaları (.h5, .tflite)
* `data/`: Beslenme rehberini içeren JSON dosyası
* `scripts/`: Eğitim ve test aşamasında kullanılan Python kodları
* `outputs/`: Accuracy/Loss grafikleri ve Karmaşıklık Matrisi

## 🎓 Akademik Bağlam
Bu çalışma, Bilecik Şeyh Edebali Üniversitesi Mühendislik Fakültesi Bilgisayar Mühendisliği Bölümü tasarım çalışması olarak hazırlanmıştır.

**Yazar:** Fatmanur Akyol  
**Yıl:** 2025-2026
