# pip install mcp 前提
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run():
    # 1. サーバーの起動条件を指定（さっき作ったファイルを指定）
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_clock_server.py"], # サーバーのファイル名
    )

    # 2. サーバーに接続する
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初期化
            await session.initialize()

            # 3. 使えるツールの一覧を取得して表示してみる
            tools = await session.list_tools()
            print("--- 利用可能なツール ---")
            for tool in tools.tools:
                print(f"名前: {tool.name}")
                print(f"説明: {tool.description}")

            # 4. 実際にツールを呼び出してみる
            print("\n--- ツールを実行中... ---")
            result = await session.call_tool("get_current_time", arguments={})
            print(f"結果: {result.content[0].text}")

if __name__ == "__main__":
    asyncio.run(run())