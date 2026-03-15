from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import datetime
import json
import chromadb
import os
from chromadb.utils import embedding_functions

# ==========================================
# 1. 環境設定とパス解決
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DATA_PATH = os.path.join(BASE_DIR, "chroma_db")

DEBUG_MODE = True
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# ==========================================
# 2. 初期化 (FastAPI / ChromaDB)
# ==========================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
emb_fn = embedding_functions.DefaultEmbeddingFunction()

# ==========================================
# 3. ユーティリティ関数
# ==========================================

def get_collection(persona_name: str):
    """
    ペルソナ名に基づいてChromaDBのコレクション（箱）を選択・作成する。
    """
    # ファイル名からベース名を取得 (yuki.txt -> yuki)
    name = os.path.splitext(persona_name)[0] if persona_name else "default"
    # ChromaDBの命名規則(3-63文字)に合わせる
    safe_name = f"mem_{name}"[:63]
    
    return chroma_client.get_or_create_collection(name=safe_name, embedding_function=emb_fn)

def load_persona_file(filename: str):
    if not filename: return None
    target_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            if DEBUG_MODE: print(f"--- [❌ Persona Read Error: {e}] ---")
    return None

def get_reference_block(query: str, persona: str):
    """
    指定されたペルソナ専用の箱から記憶を呼び出す。
    """
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    memory_text = ""
    
    # 適切なコレクションを取得
    target_col = get_collection(persona)
    
    if query and DEBUG_MODE:
        print(f"--- [🔍 Search in '{target_col.name}' trigger: '{query}'] ---")

    try:
        results = target_col.query(query_texts=[query], n_results=1)
        if results['documents'] and len(results['documents'][0]) > 0:
            memory_text = results['documents'][0][0]
            if DEBUG_MODE:
                print(f"--- [✅ Memory Found in '{target_col.name}'] ---\n{memory_text[:100]}...\n--------------------------")
    except Exception as e:
        if DEBUG_MODE: print(f"--- [⚠️ Search Skip: {e}] ---")

    block = f"\n\n### 補足参照データ（過去の事実として考慮）\n"
    block += f"- 現在時刻: {now}\n"
    if memory_text:
        block += f"- このキャラクターとの過去の対話:\n{memory_text}\n"
    block += "###\n"
    return block

def save_to_memory(user_msg: str, ai_msg: str, persona_name: str):
    """
    指定されたペルソナ専用の箱に会話を保存。
    """
    timestamp = datetime.datetime.now().isoformat()
    ai_label = os.path.splitext(persona_name)[0] if persona_name else "Assistant"
    
    # 適切なコレクションを取得
    target_col = get_collection(persona_name)
    
    combined_text = f"[User]: {user_msg}\n[{ai_label}]: {ai_msg}"
    
    if DEBUG_MODE:
        print(f"--- [💾 Memory Saved to '{target_col.name}'] ---")
    
    target_col.add(
        documents=[combined_text],
        metadatas=[{"ts": timestamp}],
        ids=[timestamp]
    )

# ==========================================
# 4. メインエンドポイント
# ==========================================

@app.post("/api/chat")
async def chat_relay(request: Request, background_tasks: BackgroundTasks, persona: str = None):
    body = await request.json()
    messages = body.get("messages", [])
    if not messages: return JSONResponse(content={"error": "No messages"}, status_code=400)

    user_query = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
    external_content = load_persona_file(persona)
    
    # 記憶の取得時にも persona を渡す
    ref_block = get_reference_block(user_query, persona)

    if messages[0]["role"] == "system":
        source = f"EXTERNAL ({persona})" if external_content else "OLLAMA_DEFAULT"
        base = external_content if external_content else messages[0]["content"]
        messages[0]["content"] = base + ref_block
    else:
        source = "NEW_SYSTEM"
        messages.insert(0, {"role": "system", "content": (external_content or "") + ref_block})

    if DEBUG_MODE: print(f"--- [🚀 Role: {source}] ---")

    async def event_generator():
        full_ai_response = ""
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/chat", json=body) as response:
                async for chunk in response.aiter_bytes():
                    try:
                        decoded_chunk = chunk.decode("utf-8")
                        json_data = json.loads(decoded_chunk)
                        if "message" in json_data and "content" in json_data["message"]:
                            full_ai_response += json_data["message"]["content"]
                    except: pass
                    yield chunk
        
        if full_ai_response:
            background_tasks.add_task(save_to_memory, user_query, full_ai_response, persona)

    return StreamingResponse(event_generator(), media_type="application/x-javascript")

# 中継用エンドポイント (tags, show)
@app.get("/api/tags")
async def get_tags():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        return JSONResponse(content=response.json(), status_code=response.status_code)

@app.post("/api/show")
async def show_model(request: Request):
    body = await request.json()
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{OLLAMA_BASE_URL}/api/show", json=body)
        return JSONResponse(content=response.json(), status_code=response.status_code)