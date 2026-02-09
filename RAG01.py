import numpy as np
from llama_cpp import Llama

# 1. 前回のエンベディング用設定（Vulkan使用）
MODEL_PATH = "./Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, embedding=True, verbose=False, n_gpu_layers=35)

# --- ここからRAGの基本 ---

# AIに教えたい「自前の知識（本棚）」
knowledge_base = [
    "司さんは、週末に房総半島へドライブに行くのが好きです。",
    "司さんは、濃いめのラーメンよりも、あっさりした塩ラーメンを好みます。",
    "司さんの家には、ユキという名前のAIアシスタントがいます。",
    "司さんは、ドライブでお昼ご飯にラーメンを食べることが好きです。"
]

def get_vector(text):
    e = llm.embed(text)
    return np.mean(e, axis=0)

def get_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 司さんの質問
user_query = "司さんはドライブでランチはなにを食べたがるかな？"

print(f"質問: {user_query}\n")

# ステップ1: 質問をベクトル化する
query_vec = get_vector(user_query)

# ステップ2: 本棚の中身を全部調べて、一番近いもの（類似度が高いもの）を探す
best_score = -1
best_info = ""

for info in knowledge_base:
    info_vec = get_vector(info)
    score = get_similarity(query_vec, info_vec)
    
    print(f"確認中: 「{info}」 (スコア: {score:.4f})")
    
    if score > best_score:
        best_score = score
        best_info = info

print(f"\n★検索結果: ユキに伝えるべきヒントはこれです！\n「{best_info}」")