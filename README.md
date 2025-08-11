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

## 📂 Veri Kaynakları
- Haberler, belirli aralıklarla seçili Türkçe haber sitelerinden otomatik olarak çekilmektedir
- Metinler embedding sonrası Pinecone’a aktarılmaktadır
- RAG mekanizması ile hem **veritabanı** hem **LLM** bilgisi kullanılır

## 🛠️ Teknik Detaylar
- **Embedding Modeli:** `intfloat/multilingual-e5-large`
- **Dil Modeli:** Fine-tuned `ytu-ce-cosmos/turkish-gpt2-large`
- **Vektör Veritabanı:** Pinecone
- **Çalışma Ortamı:** Hugging Face Spaces (Gradio tabanlı arayüz)

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

