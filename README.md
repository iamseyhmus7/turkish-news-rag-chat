📰 RAG-Sondakika-Haber
RAG-Sondakika-Haber, GPT-2 Large modelinin tamamen Türkçe veri seti ile full fine-tuning yöntemiyle eğitilerek geliştirilmiş, hem genel sorulara hem de gerçek zamanlı son dakika haberlerine yanıt verebilen bir yapay zeka projesidir.

🚀 Özellikler
Türkçe GPT-2 Large: Tam veri seti ile sıfırdan fine-tune edilerek Türkçe dil yapısını öğrenmiştir.

RAG (Retrieval-Augmented Generation) entegrasyonu sayesinde:

Genel bilgi sorgularına yanıt verebilir.

Son dakika haberleri konusunda güncel bilgiler sunar.

Otomatik Haber Toplama:

Belirli haber sitelerinden (örn. son dakika kategorileri) veriler çekilir.

İçerikler chunk sistemi ile parçalara ayrılır.

Pinecone.io Vektör Veritabanı:

Haber chunk’ları embedding yapılarak Pinecone VDB’de saklanır.

Sorgu geldiğinde en alakalı haber parçaları getirilir.

🛠 Teknolojiler
Model: GPT-2 Large (full fine-tuning, Türkçe veri seti)

VDB: Pinecone.io

Embedding: multilingual-e5-large

RAG Pipeline: Özel chunklama + vektör arama

Scraper: Python + BeautifulSoup / Requests (haber siteleri)

📌 Çalışma Mantığı
Haber Toplama → Web scraper ile son dakika haberleri çekilir.

Chunklama → Haber içerikleri anlamlı parçalara bölünür.

Embedding & Pinecone → Her chunk embedding modeli ile vektöre dönüştürülerek Pinecone’a kaydedilir.

Sorgu Cevaplama → Kullanıcı sorusu embedding yapılır, en alakalı chunk’lar bulunur.

LLM Cevabı → GPT-2 Large modeli, hem kendi bilgisini hem de haber chunk’larını kullanarak yanıt üretir.
