import numpy as np
from llama_cpp import Llama

# 1. モデルの準備
MODEL_PATH = "./Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, embedding=True, verbose=False, n_gpu_layers=35)

def get_vector(text):
    # 文章をベクトル化
    e = llm.embed(text)
    # eが(トークン数, 5120)の形なので、トークン方向に平均をとって(5120,)にする
    # これを「平均プーリング」と呼びます
    return np.mean(e, axis=0)

def get_similarity(v1, v2):
    # コサイン類似度を計算
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 2. 比較したい文章
text_a = "パスタを食べに行こう"
text_b = "今日は麺類の気分だな"
text_c = "明日は仕事で会議がある"

# 3. 意味の「重心」を取得
vec_a = get_vector(text_a)
vec_b = get_vector(text_b)
vec_c = get_vector(text_c)

# 4. 結果表示
print(f"「{text_a}」 と 「{text_b}」 の似ている度: {get_similarity(vec_a, vec_b):.4f}")
print(f"「{text_a}」 と 「{text_c}」 の似ている度: {get_similarity(vec_a, vec_c):.4f}")