# pip install mcp 前提
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run():
    # 1. サーバー（時計）の起動設定
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_clock_server.py"], 
    )

    print("（システム：MCPサーバーを起動して接続します...）")
    
    # 2. サーバーと接続
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初期化（「握手」のようなもの）
            await session.initialize()

            # --- STEP 1: MCPサーバーから「道具の生データ」を取得 ---
            tools_response = await session.list_tools()
            print("\n=== [STEP 1] MCPから取得したツールの生データ ===")
            for tool in tools_response.tools:
                print(f"名前: {tool.name}")
                print(f"説明文（docstringより）: {tool.description}")
                print(f"引数の構造: {tool.inputSchema}")

            # --- STEP 2: ブローカーによる「翻訳（プロンプト化）」 ---
            # ここが、司さんの仰る「LLMに伝えるための形」を作る部分です
            print("\n=== [STEP 2] LLM（Mistral）へ送る『お品書き』への変換 ===")
            
            mistral_format_tools = []
            for tool in tools_response.tools:
                # サーバーの情報を、Mistralが期待するJSON構造に詰め替える
                mistral_format_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            
            # Mistralの専用タグ [AVAILABLE_TOOLS] で囲む
            prompt_for_yuki = f"[AVAILABLE_TOOLS]\n{json.dumps(mistral_format_tools, indent=2, ensure_ascii=False)}\n[/AVAILABLE_TOOLS]"
            
            print("【生成されたプロンプト】")
            print(prompt_for_yuki)
            print("\n※この文字列が、実際の会話の裏側でシステムプロンプトと一緒にLLMへ送られます。")

            # --- STEP 3: 実行テスト（念のため） ---
            print("\n=== [STEP 3] 実際にツールを実行してみる ===")
            result = await session.call_tool("get_current_time", arguments={})
            print(f"実行結果: {result.content[0].text}")

if __name__ == "__main__":
    asyncio.run(run())