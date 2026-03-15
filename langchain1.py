from langchain_ollama import ChatOllama
from langchain_core.tools import tool

# 1. ユキの「手足」となる道具を定義する（これだけでAI用の仕様書が自動生成されます）
@tool
def get_current_time():
    """現在の日時を返します。時刻を確認したい時に使ってください。"""
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 2. ユキ（Mistral）の準備
# Ollamaを通じて、ツールが使えるモデルを呼び出します
llm = ChatOllama(model="yuki-chat")

# 3. ユキに道具を「持たせる（Bind）」
# これでユキは「あ、私には時計があるんだ」と認識します
llm_with_tools = llm.bind_tools([get_current_time])

# 4. 司さんが話しかける
response = llm_with_tools.invoke("ユキさん、今って何時かな？")

# 5. ユキの「意志」を確認
print(f"ユキのツール呼び出し: {response.tool_calls}")