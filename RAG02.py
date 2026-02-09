import numpy as np
from llama_cpp import Llama

# 1. モデルの準備
MODEL_PATH = "./Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf"
# embedding=Trueにすることで、検索と会話の両方ができるようになります
llm = Llama(model_path=MODEL_PATH, embedding=True, n_gpu_layers=35, n_ctx=8192, verbose=False)

# 本棚（ユキの知識）
knowledge_base = [
    "司さんは、週末に房総半島へドライブに行くのが好きです。",
    "司さんは、あっさりした塩ラーメンを好みます。",
    "司さんの家には、ユキという名前のAIアシスタントがいます。",
    "司さんは、ドライブでお昼ご飯にラーメンを食べることが好きです。"
]

def get_vector(text):
    e = llm.embed(text)
    return np.mean(e, axis=0)

def get_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def chat_with_yuki_rag(user_input):
    # --- R: Retrieval（検索） ---
    query_vec = get_vector(user_input)
    # 一番似ているメモを探す
    best_info = knowledge_base[0]
    best_score = -1
    for info in knowledge_base:
        score = get_similarity(query_vec, get_vector(info))
        if score > best_score:
            best_score = score
            best_info = info

    if best_score < 0.5:
    # 似ている度合いが低いなら、参考情報は渡さない
        best_info = "（特になし）"
    # --- A: Augmentation（拡張） ---
    # ここが「カンニングペーパー」の作成現場です！
    # システムプロンプトの中に、検索した知識を「参考情報」として無理やり差し込みます。
    prompt = f"""[SYSTEM_PROMPT]
あなたの名前はユキ、20代女性です。丁寧な敬語で話してください。
以下の【参考情報】は、あなたが思い出した「司さんに関する記憶」です。
これを知っている前提で、自然に会話してください。

【参考情報】: {best_info}
[/SYSTEM_PROMPT]
[INST] {user_input} [/INST]"""

    # --- G: Generation（生成） ---
    output = llm(prompt, max_tokens=256, stop=["[/INST]", "</s>"])
    return output["choices"][0]["text"].strip()

# テスト
if __name__ == "__main__":
    print("ユキ: こんにちは、司さん！")
    while True:
        u_input = input("あなた: ")
        if u_input.lower() in ["exit", "quit", "bye"]: break
        
        response = chat_with_yuki_rag(u_input)
        print(f"ユキ: {response}")