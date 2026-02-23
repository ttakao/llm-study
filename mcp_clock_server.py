# pip install mcp 前提
from mcp.server.fastmcp import FastMCP
from datetime import datetime

#1 MCPサーバーインスタンス
mcp = FastMCP("Yuki-Clock")

#2 Toolの登録
@mcp.tool()
def get_current_time() -> str:
    """現在の日時を[YYYY年MM月DD日 HH時MM分SS]の形式で返します"""
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H時%M分%S秒")
    
if __name__ == "__main__":
    mcp.run()