import os
import asyncio
import re
import yaml
import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from pathlib import Path
from dotenv import load_dotenv

# === CONFIG LOAD ===
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
config = load_config()

# === ENV LOAD ===
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
if not all([PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME]):
    raise ValueError("Pinecone ortam değişkenleri eksik!")

# === PINECONE CONNECT ===
pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone_client.Index(PINECONE_INDEX_NAME)

# === MODEL LOAD ===
MODEL_PATH = "iamseyhmus7/GenerationTurkishGPT2_final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

embedder = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

def clean_text_output(text: str) -> str:
    # ... Aynı şekilde bırakabilirsin ...
    text = re.sub(r"^(Sadece doğru, kısa ve açık bilgi ver\.? Ekstra açıklama veya kaynak ekleme\.?)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^.*?(Metin:|output:|Cevap:)", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^(Aşağıdaki haber.*|Yalnızca olay özeti.*|Cevapta sadece.*|Metin:|output:|Cevap:)", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"(Detaylı bilgi için.*|Daha fazla bilgi için.*|Wikipedia.*|Kaynak:.*|https?://\S+)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\- ", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+[\.\)]?\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_embedding(text: str, max_length: int = 512) -> np.ndarray:
    formatted = f"query: {text.strip()}"[:max_length]
    return embedder.encode(formatted, normalize_embeddings=True)

def pinecone_query(query: str, top_k: int) -> list:
    query_embedding = get_embedding(query)
    result = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
    matches = result.get("matches", [])
    output = []
    for m in matches:
        text = m.get("metadata", {}).get("text", "").strip()
        url = m.get("metadata", {}).get("url", "")
        if text:
            output.append((text, url))
    return output

async def retrieve_sources_from_pinecone(query: str, top_k: int = None):
    top_k = top_k or config["pinecone"]["top_k"]
    output = pinecone_query(query, top_k)
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
        f"input: {question}\noutput:" "Sadece doğru, kısa ve açık bilgi ver. Ekstra açıklama veya kaynak ekleme."
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
    match = re.search(r"output:(.*)", output, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    if "Cevap:" in output:
        return output.split("Cevap:")[-1].strip()
    return output.strip()

def cut_at_last_period(text):
    """Metni son noktaya kadar keser, sonrasında kalan eksik cümleleri atar."""
    last_period = text.rfind(".")
    if last_period != -1:
        return text[:last_period+1].strip()
    return text.strip()

# ... (devamı aynı)

async def selfrag_agent(question: str):
    # 1. VDB cevabı ve kaynak url
    result = await retrieve_sources_from_pinecone(question)
    vdb_paragraph = result.get("sources", "").strip()
    source_url = result.get("source_url", "")

    # 2. Model cevabı
    model_paragraph = await generate_model_response(question)
    model_paragraph = extract_self_answer(model_paragraph)

    # 3. Temizle
    vdb_paragraph = clean_text_output(vdb_paragraph)
    model_paragraph = clean_text_output(model_paragraph)

    # --- NOKTA KONTROLÜ EKLENDİ ---
    vdb_paragraph = cut_at_last_period(vdb_paragraph)
    model_paragraph = cut_at_last_period(model_paragraph)
    # -----------------------------

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
        if model_score > 1.5 * vdb_score:
            best_idx = 1
        else:
            best_idx = 0
    else:
        best_idx = int(np.argmax(scores))

    final_answer = candidates[best_idx]
    final_source_url = candidate_urls[best_idx]

    # --- SON NOKTA KONTROLÜ FINAL CEVAPTA DA ---
    final_answer = cut_at_last_period(final_answer)
    # -------------------------------------------

    return {
        "answer": final_answer,
        "source_url": final_source_url
    }

def gradio_chat(message, history):
    try:
        result = asyncio.run(selfrag_agent(message))
        response = result["answer"]
        if result.get("source_url"):
            response += f"\n\n[Daha fazla bilgi için tıkla]({result['source_url']})"
    except Exception as e:
        response = f"Hata oluştu: {e}"
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
             Hem genel bilgi, hem en güncel haberleri RAG teknolojisiyle birleştiren asistan.<br>
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
            gr.Markdown("#### Son Eklenen Haberler")  # İstersen burayı dinamik yapabilirsin
            haberler = gr.Markdown("• Haber başlığı 1\n• Haber başlığı 2\n• Haber başlığı 3")
    msg.submit(gradio_chat, [msg, chatbot], [msg, chatbot])
    send_btn.click(gradio_chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
demo.launch()
