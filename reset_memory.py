import chromadb
import os
import sys

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHROMA_DATA_PATH = os.path.join(BASE_DIR, "chroma_db")

    if len(sys.argv) < 2:
        print("\n❌ エラー: 消去したいペルソナ（またはファイル名）を指定してください。")
        print("例: python reset_memory.py kaori.txt")
        print("例: python reset_memory.py kaori")
        return

    input_name = sys.argv[1]
    
    # --- ここがポイント：入力を整理する ---
    # 1. パスが含まれていてもファイル名だけ抽出
    filename = os.path.basename(input_name)
    # 2. 拡張子 (.txtなど) があれば取り除く
    persona_name = os.path.splitext(filename)[0]
    # 3. DB上の名前 (mem_kaori) を作成
    collection_name = f"mem_{persona_name}"

    if not os.path.exists(CHROMA_DATA_PATH):
        print(f"❌ エラー: データベースが存在しません。")
        return

    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

    try:
        collections = client.list_collections()
        existing_names = [c.name for c in collections]

        print(f"--- [📊 現在のデータベース状況] ---")
        for name in existing_names:
            print(f"・ {name}")
        print("----------------------------------")

        if collection_name in existing_names:
            print(f"⚠️  警告: '{collection_name}' の全記憶を消去します...")
            client.delete_collection(name=collection_name)
            print(f"✅ 完了: '{collection_name}' をリセットしました。")
        else:
            print(f"❌ 未発見: '{collection_name}' という箱はありません。")
            print(f"※ 入力 '{input_name}' は '{collection_name}' として探されました。")

    except Exception as e:
        print(f"❌ エラー発生: {e}")

if __name__ == "__main__":
    main()