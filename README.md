# BANKA MÜŞTERİ ŞİKAYETLERİNİ TAHMİN ETME MODELİ

## 📋 PROJE ÖZETİ

Bu proje, banka müşteri şikayetlerini otomatik olarak kategorilere ayırmak için geliştirilmiş bir makine öğrenmesi sistemidir. Türkçe metin analizi kullanarak şikayetleri farklı kategorilere sınıflandırır ve boş kategorileri otomatik olarak doldurur.

## 🎯 AMAÇ

- **Otomatik Kategorizasyon**: Müşteri şikayetlerini metin analizi ile kategorilere ayırma
- **Veri Tamamlama**: Boş kategorileri otomatik doldurma
- **Performans Analizi**: Model performansını değerlendirme ve görselleştirme
- **Çoklu Model Desteği**: Farklı makine öğrenmesi yaklaşımlarını karşılaştırma

## 📁 DOSYA YAPISI VE AÇIKLAMALARI

### 🔧 Ana Dosyalar

#### 1. `bankamodel.ipynb`
**Amaç**: Jupyter notebook formatında interaktif model geliştirme ve analiz

**İçerik**:
- **Hücre 0**: Gerekli kütüphanelerin import edilmesi
- **Hücre 1**: Türkçe metin normalizasyonu ve stopwords tanımlama
- **Hücre 2**: Pipeline oluşturma fonksiyonları
- **Hücre 3**: Model eğitimi (SVM + TF-IDF)
- **Hücre 4**: NaN kategorileri doldurma
- **Hücre 5**: Model değerlendirme ve görselleştirme
- **Hücre 6**: WordCloud analizi
- **Hücre 7**: Tek tuşla çalışan ana fonksiyon

**Kullanım**: Jupyter notebook ortamında hücreleri sırayla çalıştırın

#### 2. `svm_fill_categories.py`
**Amaç**: Standalone Python scripti - otomatik kategori doldurma

**Özellikler**:
- **Çoklu Model Desteği**: TF-IDF, SBERT, BERT modelleri
- **Ensemble Yaklaşımı**: Word + Char TF-IDF kombinasyonu
- **Kural Tabanlı Override**: Anahtar kelime bazlı düzeltme sistemi
- **Pseudo-labeling**: Yüksek güvenilirlikli tahminleri eğitim verisine ekleme
- **Otomatik Hiperparametre Optimizasyonu**: Grid search ile en iyi parametreleri bulma
- **Görselleştirme**: Confusion matrix, sınıf dağılımı, F1 skorları

**Kullanım**: `python svm_fill_categories.py`

### 📊 Veri Dosyaları

#### 3. `musteri sikayetleri.xlsx`
**Amaç**: Girdi verisi - müşteri şikayetleri ve kategorileri

**Sütunlar**:
- `tarih`: Şikayet tarihi
- `yorum`: Müşteri şikayet metni
- `kategori`: Şikayet kategorisi (bazıları boş olabilir)

#### 4. `musteri_sikayetleri_tahminli.xlsx`
**Amaç**: Çıktı verisi — tahmin edilen kategorilerle birlikte

**Ek Sütunlar**:
- `tahmin_guveni`: Model tahmin güvenilirliği (0-1 arası)
- `tahmin_model_accuracy`: Model doğruluk oranı

> Not (Yeni): Script artık Top-2/Top-3 önerilerini de üretir ve ayrı bir dosyaya yazar (aşağıya bakınız).

#### 4b. `musteri_sikayetleri_tahminli_top3.xlsx` (Yeni)
**Amaç**: İlk tahminle birlikte alternatif 2. ve 3. seçenekleri de içeren çıktı

**Ek Sütunlar**:
- `tahmin_top1`, `tahmin_top1_conf`
- `tahmin_top2`, `tahmin_top2_conf`
- `tahmin_top3`, `tahmin_top3_conf`

### 📈 Görselleştirme Dosyaları

#### 5. `confusion_matrix_validation.png`
**Amaç**: Model performansının confusion matrix görselleştirmesi
**İçerik**: Doğru ve yanlış tahminlerin matris gösterimi

#### 6. `sinif_dagilimi.png`
**Amaç**: Veri setindeki sınıf dağılımının görselleştirmesi
**İçerik**: Her kategorideki şikayet sayısının bar grafiği

#### 7. `per_class_f1.png`
**Amaç**: Her sınıf için F1 skorlarının görselleştirmesi
**İçerik**: Kategori bazında model performansının bar grafiği

#### 8. `wordcloud_overall.png`
**Amaç**: Tüm şikayetlerdeki kelime bulutunun görselleştirmesi
**İçerik**: En sık kullanılan kelimelerin görsel temsili

### 📋 Rapor Dosyaları

#### 9. `model_metrics.json`
**Amaç**: Model performans metriklerinin JSON formatında saklanması
**İçerik**: Accuracy, F1 skorları, sınıflandırma raporu

### 📂 Eski Veri Klasörü

#### 10. `eski veri/predictions_all_models_nan_only.xlsx`
**Amaç**: Önceki model çalıştırmalarından elde edilen tahminler
**İçerik**: Farklı modellerin NaN satırlar için yaptığı tahminler

## 🚀 KULLANIM KILAVUZU

### Hızlı Başlangıç

1. **Gereksinimlerin Kurulumu**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn wordcloud openpyxl
   ```

2. **Veri Hazırlığı**:
   - `musteri sikayetleri.xlsx` dosyasını proje klasörüne koyun
   - Dosyada `yorum` ve `kategori` sütunlarının olduğundan emin olun

3. **Model Çalıştırma**:
   ```bash
   python svm_fill_categories.py
   ```

4. **Sonuçları İnceleme**:
   - `musteri_sikayetleri_tahminli.xlsx` dosyasını kontrol edin
   - Görselleştirme dosyalarını inceleyin
   - `model_metrics.json` dosyasından performans metriklerini okuyun

### Jupyter Notebook Kullanımı

1. **Notebook'u Açın**:
   ```bash
   jupyter notebook bankamodel.ipynb
   ```

2. **Hücreleri Çalıştırın**:
   - Hücreleri sırayla çalıştırın (Shift + Enter)
   - Hücre 7'deki `train_fill_save()` fonksiyonunu çağırın

3. **Sonuçları Analiz Edin**:
   - Görselleştirmeleri inceleyin
   - Model performansını değerlendirin

## 🔧 TEKNİK DETAYLAR

### Kullanılan Teknolojiler

- **Python 3.7+**: Ana programlama dili
- **scikit-learn**: Makine öğrenmesi algoritmaları
- **pandas**: Veri manipülasyonu
- **numpy**: Sayısal işlemler
- **matplotlib/seaborn**: Görselleştirme
- **wordcloud**: Kelime bulutu oluşturma
- **transformers**: BERT modelleri (opsiyonel)

### Model Yaklaşımları

1. **TF-IDF + LinearSVC**:
   - Metinleri TF-IDF ile vektörize eder
   - Linear Support Vector Classifier ile sınıflandırır
   - Hızlı ve etkili

2. **Ensemble TF-IDF**:
   - Word-based ve char-based TF-IDF'i birleştirir
   - Daha iyi performans sağlar

3. **SBERT (Sentence-BERT)**:
   - Cümle seviyesinde embedding'ler
   - Daha anlamlı metin temsili

4. **BERT Fine-tuning**:
   - Türkçe BERT modelini fine-tune eder
   - En yüksek performans (daha yavaş)

### Özellik Mühendisliği

- **Türkçe Metin Normalizasyonu**: Karakter dönüşümleri, URL/HTML temizleme
- **Stopwords Filtreleme**: Gereksiz kelimelerin çıkarılması
- **N-gram Özellikleri**: Kelime ve karakter n-gram'ları
- **Sınıf Ağırlıklandırma**: Dengesiz veri setlerini ele alma

## 📊 PERFORMANS METRİKLERİ

### Değerlendirme Kriterleri

- **Accuracy**: Genel doğruluk oranı
- **Macro-F1**: Sınıf bazında ortalama F1 skoru
- **Weighted-F1**: Sınıf büyüklüğüne göre ağırlıklı F1 skoru
- **Confusion Matrix**: Detaylı sınıflandırma performansı

### Beklenen Performans

- **TF-IDF + SVM**: %75-85 accuracy
- **Ensemble TF-IDF**: %80-90 accuracy
- **SBERT**: %85-95 accuracy
- **BERT Fine-tuning**: %90-98 accuracy

## 🔍 SORUN GİDERME

### Yaygın Sorunlar

1. **Excel Dosyası Bulunamıyor**:
   - Dosya adının `musteri sikayetleri.xlsx` olduğundan emin olun
   - Dosyanın proje klasöründe olduğunu kontrol edin

2. **Sütun Bulunamıyor**:
   - Excel dosyasında `yorum` ve `kategori` sütunlarının olduğunu kontrol edin
   - Sütun isimlerinin doğru yazıldığından emin olun

3. **Model Performansı Düşük**:
   - Veri kalitesini kontrol edin
   - Etiketli veri miktarını artırın
   - Hiperparametreleri optimize edin

4. **Bellek Hatası**:
   - Veri setini küçültün
   - `max_features` parametresini azaltın
   - Daha az iterasyon kullanın

## 📝 LİSANS

Bu proje eğitim ve araştırma amaçlı geliştirilmiştir.




