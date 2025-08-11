# 📰 RAG News Chatbot

Bu proje, **Retrieval-Augmented Generation (RAG)** yaklaşımını kullanarak **Türkçe haberler** üzerinde soru-cevap ve sohbet işlevi sunan bir yapay zekâ chatbotudur.  
Sistem, Pinecone vektör veritabanında saklanan haber içeriklerinden ilgili bilgileri çekip **fine-tuned GPT-2** modelini kullanarak yanıt üretir.  
Canlı olarak denemek için [🌐 Hugging Face Space - RAG News Chatbot](https://huggingface.co/spaces/iamseyhmus7/turkish-news-rag-chat) sayfasını ziyaret edebilirsiniz.

## 🚀 Özellikler
- Türkçe **son dakika haberleri** ile güncel cevaplar
- Pinecone vektör veritabanı ile hızlı içerik sorgulama
- **multilingual-e5-large** modeli ile embedding işlemleri
- Fine-tuned GPT-2 ile doğal ve bağlama uygun cevaplar
- Web tabanlı sohbet arayüzü (Hugging Face Spaces üzerinde)
- **Docker ile tam otomasyon** – Her saat başı haber sitelerinden yeni verilerin otomatik çekilmesi, işlenmesi ve Pinecone’a aktarılması

## 📂 Veri Kaynakları
- Haberler, **Docker tabanlı zamanlayıcı** ile her saat başı seçili Türkçe haber sitelerinden otomatik olarak çekilmektedir
- Çekilen metinler embedding işleminden geçirilerek Pinecone vektör veritabanına kaydedilir
- RAG mekanizması ile hem **güncel haber verisi** hem de **model bilgisi** birlikte kullanılır

## 🛠️ Teknik Detaylar
- **Embedding Modeli:** `intfloat/multilingual-e5-large`
- **Dil Modeli:** Fine-tuned `ytu-ce-cosmos/turkish-gpt2-large`
- **Vektör Veritabanı:** Pinecone
- **Çalışma Ortamı:** Hugging Face Spaces (Gradio tabanlı arayüz)
- **Otomasyon:** Docker container üzerinde çalışan scraper ve pipeline süreçleri
- **Veri Çekme Sıklığı:** Saatte bir (cron tabanlı otomasyon)

## 🐳 Docker Kullanımı
Projede Docker kullanılarak hem veri toplama hem de embedding + vektör veritabanı işlemleri tamamen otomatikleştirilmiştir.  
Bu sayede:
- Sistem her yeniden başlatıldığında kendini otomatik olarak çalıştırır
- Haber verileri düzenli olarak güncellenir
- Yerel ortam fark etmeksizin aynı yapı çalıştırılabilir

## 📦 Kullanım
### Hugging Face ile
```python
from transformers import pipeline

# Modeli yükle
pipe = pipeline("text-generation", model="iamseyhmus7/GenerationTurkishGPT2_final")

# Metin örneği
prompt = "Yapay zekâ gelecekte hayatımızı nasıl etkileyecek?"

# Tahmin/generasyon al
output = pipe(prompt, max_new_tokens=128)
print(output[0]['generated_text'])

