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
# ※一度登録すれば、プログラムを落としても消えません
memories = [
    "司さんは週末、房総半島へドライブに行くのが好きです。",
    "司さんはあっさりした塩ラーメンを好みます。",
    "ユキは司さんの大切なAIパートナーです。"
]

# ID（0, 1, 2...）を付けて保存
for i, text in enumerate(memories):
    collection.upsert(
        ids=[str(i)],
        embeddings=[get_vector(text)],
        documents=[text]
    )

print("--- データベースへの登録が完了しました ---")

# 4. 検索してみる
user_query = "ドライブのおすすめある？"
query_vec = get_vector(user_query)

# データベースに「近いもの上位1件を持ってきて」と頼む
results = collection.query(
    query_embeddings=[query_vec],
    n_results=1
)

# 5. 結果の取り出し
best_doc = results['documents'][0][0]
score = results['distances'][0][0] # ChromaDBは「距離」を出すので、小さいほど似ている

print(f"質問: {user_query}")
print(f"検索された記憶: {best_doc} (距離スコア: {score:.4f})")