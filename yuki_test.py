from llama_cpp import Llama
import os

# --- 設定項目 ---
# モデルファイルのパス（実際のファイル名に合わせて変更してください）
MODEL_PATH = "./Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf"
PROMPT_PATH = "./mistral_prompt.txt"

# SYSTEMプロンプトの読み取り
if os.path.exists(PROMPT_PATH):
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_content = f.read()
else:
    system_content = "あなたはユキ、２０題の日本人女性です。"

# モデルの初期化
llm = Llama(
    model_path = MODEL_PATH,
    n_gpu_layers=35,
    n_ctx=8192,
    n_threads=8,
    verbose=True
)

def chat_with_yuki(user_text):
    prompt = f"""[SYSTEM_PROMPT] {system_content} [/SYSTEM_PROMPT] [INST] {user_text} [/INST]"""

    output = llm(
        prompt,
        max_tokens = 256,
        temperature = 0.8,
        top_p = 0.9, 
        repeat_penalty = 1.2,
        stop = ["[/INST]", "</s>"]
    )
    return output["choices"][0]["text"].strip()
                  
if __name__ == "__main__":
    print("ユキ：こんにちは")
    while True:
        u_input = input("あなた： ")
        if u_input.lower() in ["exit", "quit", "bye"]: break

        response = chat_with_yuki(u_input)
        print( f"ユキ: {response}" )
        