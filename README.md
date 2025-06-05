# 🌾 Derin Öğrenme ile Pirinç Yaprağı Hastalıklarının Sınıflandırılması (2024–2025)

Bu repoda, pirinç yaprağı hastalıklarını sınıflandırmak amacıyla önceden eğitilmiş CNN mimarileri kullanılarak gerçekleştirilmiş bir derin öğrenme projesi yer almaktadır. Çalışma, 2024–2025 Bahar Dönemi **Derin Öğrenme** dersi kapsamında gerçekleştirilmiştir.

## 📁 Veri Kümesi

Kullanılan veri seti **Kaggle** üzerinden alınmış **Rice Leaf Disease** veri kümesidir. Toplamda **5932 görüntü** içerir ve 4 farklı hastalık sınıfını kapsamaktadır:
- Bacterial Blight
- Blast
- Brown Spot
- Tungro

📎 [Kaggle Veri Kümesi](https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image)

Görseller **224x224** boyutuna ölçeklendirilmiş ve veri seti **%80 eğitim** – **%20 doğrulama** olacak şekilde ayrılmıştır.

## 🧠 Kullanılan Modeller

Aşağıdaki 4 adet önceden eğitilmiş CNN modeli kullanılmıştır:
- ✅ MobileNetV2
- ✅ ResNet50
- ✅ VGG16
- ✅ Xception

Her bir model için 8 farklı hiperparametre kombinasyonu test edilmiştir:
- Dropout oranı: 0.2 / 0.4
- Dense katman sayısı: 64 / 128 / 256
- Data Augmentation: Açık / Kapalı

## 📊 Değerlendirme Metrikleri

Modeller aşağıdaki ölçütlere göre değerlendirilmiştir:
- Doğruluk (Accuracy)
- Kayıp (Loss)
- Confusion Matrix (Karmaşıklık Matrisi)
- Sınıflandırma Raporu (Precision, Recall, F1-score)

🎨 Görselleştirmeler:
- Epoch bazlı doğruluk ve kayıp grafikleri
- Normalleştirilmiş karmaşıklık matrisi
- Yanlış sınıflandırılmış görüntüler
