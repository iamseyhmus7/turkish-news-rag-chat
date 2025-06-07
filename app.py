import os
import asyncio
import logging
import re
import yaml
import torch
import numpy as np
import gradio as gr
from functools import lru_cache
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict

# === LOGGING ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIG LOAD ===
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
def load_config() -> Dict:
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Konfigürasyon dosyası yüklenemedi: {e}")
        return {
            "pinecone": {"top_k": 10, "rerank_top": 5, "batch_size": 32},
            "model": {"max_new_tokens": 50, "temperature": 0.7},
            "cache": {"maxsize": 100}
        }
config = load_config()

# === LOAD ENV ===
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


# === PINECONE CONNECT ===
pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
try:
    index = pinecone_client.Index(PINECONE_INDEX_NAME)
    index_stats = index.describe_index_stats()
    logger.info(f"Pinecone index stats: {index_stats}")
except Exception as e:
    logger.error(f"Pinecone bağlantı hatası: {e}")
    raise

# === MODEL LOAD ===
MODEL_PATH = "iamseyhmus7/GenerationTurkishGPT2_final"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model {MODEL_PATH} Hugging Face Hub'dan yüklendi, cihaz: {device}")
except Exception as e:
    logger.error(f"Model yükleme hatası: {e}")
    raise


# === EMBEDDING MODELS ===
embedder = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
logger.info("Embedding ve reranking modelleri yüklendi")

# === FASTAPI ===
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

class QuestionRequest(BaseModel):
    query: str

def clean_text_output(text: str) -> str:
    """
    Tüm prompt, komut, yönerge, link ve gereksiz açıklamaları temizler.
    Sadece net, kısa yanıtı bırakır.
    """
    # Modelin başındaki yönerge/talimat cümleleri
    text = re.sub(
        r"^(Sadece doğru, kısa ve açık bilgi ver\.? Ekstra açıklama veya kaynak ekleme\.?)", 
        "", text, flags=re.IGNORECASE
    )
    # Büyük prompt ve yönergeleri sil (Metin:, output:, Cevap:)
    text = re.sub(r"^.*?(Metin:|output:|Cevap:)", "", text, flags=re.IGNORECASE | re.DOTALL)
    # Tek satırlık açıklama veya yönerge kalanlarını sil
    text = re.sub(r"^(Aşağıdaki haber.*|Yalnızca olay özeti.*|Cevapta sadece.*|Metin:|output:|Cevap:)", "", text, flags=re.IGNORECASE | re.MULTILINE)
    # 'Detaylı bilgi için', 'Daha fazla bilgi için', 'Wikipedia', 'Kaynak:', linkler vs.
    text = re.sub(r"(Detaylı bilgi için.*|Daha fazla bilgi için.*|Wikipedia.*|Kaynak:.*|https?://\S+)", "", text, flags=re.IGNORECASE)
    # Madde işaretleri ve baştaki sayı/karakterler
    text = re.sub(r"^\- ", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+[\.\)]?\s+", "", text, flags=re.MULTILINE)
    ## Model promptlarının başında kalan talimat cümlelerini sil
    text = re.sub(
        r"^(Sadece doğru, kısa ve açık bilgi ver\.? Ekstra açıklama veya kaynak ekleme\.?)", 
        "", text, flags=re.IGNORECASE
    )
    # Tekrarlı boşluklar ve baş/son boşluk
    text = re.sub(r"\s+", " ", text).strip()
    return text

@lru_cache(maxsize=config["cache"]["maxsize"])
def get_embedding(text: str, max_length: int = 512) -> np.ndarray:
    formatted = f"query: {text.strip()}"[:max_length]
    return embedder.encode(formatted, normalize_embeddings=True)

@lru_cache(maxsize=32)
def pinecone_query_cached(query: str, top_k: int) -> tuple:
    query_embedding = get_embedding(query)
    result = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
    matches = result.get("matches", [])
    output = []
    for m in matches:
        text = m.get("metadata", {}).get("text", "").strip()
        url = m.get("metadata", {}).get("url", "")
        if text:
            output.append((text, url))
    return tuple(output)

async def retrieve_sources_from_pinecone(query: str, top_k: int = None) -> Dict[str, any]:
    top_k = top_k or config["pinecone"]["top_k"]
    output = pinecone_query_cached(query, top_k)
    if not output:
        return {"sources": "", "results": [], "source_url": ""}
    # Cross-encoder ile yeniden sıralama
    sentence_pairs = [[query, text] for text, url in output]
    scores = await asyncio.to_thread(cross_encoder.predict, sentence_pairs)
    reranked = [(float(score), text, url) for score, (text, url) in zip(scores, output)]
    reranked.sort(key=lambda x: x[0], reverse=True)
    top_results = reranked[:1]
    top_texts = [text for _, text, _ in top_results]
    source_url = top_results[0][2] if top_results else ""
    return {"sources": "\n".join(top_texts), "results": top_results, "source_url": source_url}

async def generate_model_response(question: str) -> str:
    prompt = (
        f"input: {question}\noutput:"
        "Sadece doğru, kısa ve açık bilgi ver. Ekstra açıklama veya kaynak ekleme."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def extract_self_answer(output: str) -> str:
    # Eğer "output:" etiketi varsa, sonrasını al
    match = re.search(r"output:(.*)", output, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    # Eğer "Cevap:" varsa, sonrasını al
    if "Cevap:" in output:
        return output.split("Cevap:")[-1].strip()
    return output.strip()

async def selfrag_agent(question: str):
    # 1. VDB cevabı ve kaynak url
    result = await retrieve_sources_from_pinecone(question)
    vdb_paragraph = result.get("sources", "").strip()
    source_url = result.get("source_url", "")

    # 2. Model cevabı
    model_paragraph = await generate_model_response(question)
    model_paragraph = extract_self_answer(model_paragraph)

    # 3. Temizle (SADECE METİN DEĞERLERİNDE!)
    vdb_paragraph = clean_text_output(vdb_paragraph)
    model_paragraph = clean_text_output(model_paragraph)

    # 4. Cross-encoder ile skorlama
    candidates = []
    candidate_urls = []
    label_names = []
    if vdb_paragraph:
        candidates.append(vdb_paragraph)
        candidate_urls.append(source_url)
        label_names.append("VDB")
    if model_paragraph:
        candidates.append(model_paragraph)
        candidate_urls.append(None)
        label_names.append("MODEL")

    if not candidates:
        return {"answer": "Cevap bulunamadı.", "source_url": None}

    sentence_pairs = [[question, cand] for cand in candidates]
    scores = await asyncio.to_thread(cross_encoder.predict, sentence_pairs)
    print(f"VDB Skor: {scores[0]:.4f}")
    if len(scores) > 1:
        print(f"Model Skor: {scores[1]:.4f}")

    # === Seçim Kuralları ===
    if len(scores) == 2:
        vdb_score = scores[0]
        model_score = scores[1]
        # Eğer modelin skoru, VDB'nin 2 katından fazlaysa modeli döndür
        if model_score > 1.5 * vdb_score:
            best_idx = 1
        else:
            best_idx = 0
    else:
        # Sadece VDB veya model varsa, en yüksek skoru seç
        best_idx = int(np.argmax(scores))

    final_answer = candidates[best_idx]
    final_source_url = candidate_urls[best_idx]

    return {
        "answer": final_answer,
        "source_url": final_source_url
    }


def gradio_chat(message, history):
    # ... burada selfrag_agent_sync çağrısı olacak ...
    import time
    time.sleep(1)  # Sadece örnek animasyon için
    result = {"answer": "Bu bir örnek cevaptır. \n\n[Daha fazla bilgi için tıkla](https://example.com)", "source_url": "https://example.com"}
    response = result["answer"]
    history = history + [[message, response]]
    return "", history

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <div style='display: flex; align-items: center; gap: 16px; margin-bottom: 12px;'>
            <img src='https://em-content.zobj.net/source/telegram/403/globe-showing-europe-africa_1f30d.png' width='50'>
            <h1 style='margin-bottom: 0'>Türkçe Son Dakika RAG Chatbot</h1>
        </div>
        <p style='font-size: 18px; color: #666; margin-top:0'>
            🚀 Hem genel bilgi, hem en güncel haberleri RAG teknolojisiyle birleştiren asistan.<br>
            İstediğini sor, canlı haberlere ulaş!
        </p>
        """
    )
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(show_copy_button=True, height=480)
            msg = gr.Textbox(label="Bir soru yazın ve Enter'a basın", scale=2)
            with gr.Row():
                send_btn = gr.Button("Gönder")
                clear = gr.Button("Sohbeti Temizle")
        with gr.Column(scale=1):
            gr.Markdown("#### Son Eklenen Haberler")  # Buraya dilersen VDB'den başlık çekebilirsin
            haberler = gr.Markdown("• Haber başlığı 1\n• Haber başlığı 2\n• Haber başlığı 3")
    status = gr.Markdown("")

    def user_ask(message, history):
        status.update("Cevap hazırlanıyor... ⏳")
        answer = gradio_chat(message, history)
        status.update("")  # Temizle
        return answer

    msg.submit(user_ask, [msg, chatbot], [msg, chatbot])
    send_btn.click(user_ask, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
