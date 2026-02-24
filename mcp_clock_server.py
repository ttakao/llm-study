# pip install mcp 前提
from mcp.server.fastmcp import FastMCP
from datetime import datetime

# サーバーに名前をつけます
mcp = FastMCP("Yuki-Clock")

# --- ツール1: 日付のみ ---
@mcp.tool()
def get_current_date() -> str:
    """現在の日付（年・月・日）のみを取得します。
    曜日は含まれません。今日が何日か知りたい時に使います。"""
    now = datetime.now()
    return now.strftime("%Y年%m月%d日")

# --- ツール2: 時刻のみ ---
@mcp.tool()
def get_current_time() -> str:
    """現在の時刻（時・分・秒）のみを取得します。
    日付の情報は含まれません。今が何時か知りたい時に使います。"""
    now = datetime.now()
    return now.strftime("%H時%M分%S秒")

if __name__ == "__main__":
    mcp.run()