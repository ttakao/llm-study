import chromadb
import numpy as np
from llama_cpp import Llama

# 1. いつもの準備（埋め込み用）
MODEL_PATH = "./Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, embedding=True, verbose=False, n_gpu_layers=35)

def get_vector(text):
    e = llm.embed(text)
    return [float(n) for n in np.mean(e, axis=0)] # ChromaDBはfloatのリストを好みます

# 2. Vector DBの準備（'yuki_memory'という名前のフォルダに保存されます）
client = chromadb.PersistentClient(path="./yuki_memory")
collection = client.get_or_create_collection(name="memories")

# 3. 記憶をデータベースに「マージ（登録）」する

# 4. 検索してみる
user_query = "ユキさん、おはよう"
query_vec = get_vector(user_query)

# データベースに「近いもの上位1件を持ってきて」と頼む
results = collection.query(
    query_embeddings=[query_vec],
    n_results=1
)

best_doc = results['documents'][0][0] # 取り出す

# 5. LLMの呼び出しとbest_docを組み込む
prompt = f"""[SYSTEM_PROMPT]
あなたの名前はユキです。
以下の【記憶】を思い出しながら、ユーザーに挨拶してください。
【記憶】: {best_doc}
[/SYSTEM_PROMPT]
[INST] {user_query} [/INST]"""

# LLMがここで初めて「文章」を考えます
output = llm(prompt, max_tokens=128, stop=["[/INST]", "</s>"])
yuki_answer = output["choices"][0]["text"].strip()

print(f"質問: {user_query}")
print(f"ユキの返答: {yuki_answer}") # ← これがLLMが考えた言葉！