import os
import gradio as gr
import asyncio
import yaml
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from dotenv import load_dotenv
from pathlib import Path
import re

# === LOAD CONFIG ===
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
config = load_config()

# === LOAD ENV ===
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# === PINECONE ===
pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone_client.Index(PINECONE_INDEX_NAME)

# === MODEL ===
MODEL_PATH = "iamseyhmus7/GenerationTurkishGPT2_final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# === EMBEDDINGS ===
embedder = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

# === RAG AGENT ===
def clean_text_output(text: str) -> str:
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

def extract_self_answer(output: str) -> str:
    match = re.search(r"output:(.*)", output, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    if "Cevap:" in output:
        return output.split("Cevap:")[-1].strip()
    return output.strip()

def generate_model_response(question: str) -> str:
    prompt = f"input: {question}\noutput:Sadece doğru, kısa ve açık bilgi ver. Ekstra açıklama veya kaynak ekleme."
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

def selfrag_agent_sync(question: str):
    # VDB cevabı
    output = pinecone_query(question, config["pinecone"]["top_k"])
    if not output:
        vdb_paragraph, source_url = "", ""
    else:
        # Cross-encoder ile rerank
        sentence_pairs = [[question, text] for text, url in output]
        scores = cross_encoder.predict(sentence_pairs)
        reranked = sorted(zip(scores, output), key=lambda x: x[0], reverse=True)
        vdb_paragraph, source_url = reranked[0][1]

    # Model cevabı
    model_paragraph = generate_model_response(question)
    model_paragraph = extract_self_answer(model_paragraph)

    vdb_paragraph = clean_text_output(vdb_paragraph)
    model_paragraph = clean_text_output(model_paragraph)

    candidates = []
    candidate_urls = []
    if vdb_paragraph:
        candidates.append(vdb_paragraph)
        candidate_urls.append(source_url)
    if model_paragraph:
        candidates.append(model_paragraph)
        candidate_urls.append(None)
    if not candidates:
        return {"answer": "Cevap bulunamadı.", "source_url": None}
    # Cross-encoder skor
    sentence_pairs = [[question, cand] for cand in candidates]
    scores = cross_encoder.predict(sentence_pairs)
    best_idx = int(np.argmax(scores))
    final_answer = candidates[best_idx]
    final_source_url = candidate_urls[best_idx]
    return {
        "answer": final_answer,
        "source_url": final_source_url
    }

def gradio_chat(message, history):
    try:
        result = selfrag_agent_sync(message)
        response = result["answer"]
        if result.get("source_url"):
            response += f"\n\n[Daha fazla bilgi için tıkla]({result['source_url']})"
    except Exception as e:
        response = f"Hata oluştu: {e}"
    history = history + [[message, response]]
    return "", history

with gr.Blocks() as demo:
    gr.Markdown("# Türkçe Self-RAG Chatbot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Soru sorun, Enter'a basın")
    clear = gr.Button("Sohbeti Temizle")
    msg.submit(gradio_chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
demo.launch()
