# Visualize the vector DB

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import tempfile
import glob

load_dotenv()

def debug_vectorstore():
    embeddings = OpenAIEmbeddings()
    
    # ✅ Method 1: Read from the saved path file
    db_path = None
    if os.path.exists(".db_path"):
        with open(".db_path", "r") as f:
            db_path = f.read().strip()
        print(f"📁 Found saved database path: {db_path}")
    
    # ✅ Method 2: Search for ChromaDB in temp directory as backup
    if not db_path or not os.path.exists(db_path):
        print("🔍 Searching for ChromaDB in temp directory...")
        temp_dir = tempfile.gettempdir()
        chroma_dirs = glob.glob(os.path.join(temp_dir, "chroma_db_*"))
        
        if chroma_dirs:
            # Use the most recent one
            db_path = max(chroma_dirs, key=os.path.getctime)
            print(f"📁 Found ChromaDB directory: {db_path}")
        else:
            print("❌ No ChromaDB directories found!")
            return
    
    # ✅ Method 3: Also check the old "db" directory
    if not db_path or not os.path.exists(db_path):
        old_db_path = os.path.abspath("db")
        if os.path.exists(old_db_path):
            db_path = old_db_path
            print(f"📁 Using old database path: {db_path}")
    
    if not db_path or not os.path.exists(db_path):
        print("❌ No vector database found!")
        print("💡 Make sure to run /train endpoint first to create embeddings")
        return
    
    print(f"\n🔗 Connecting to database at: {db_path}")
    
    try:
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # Get all data
        data = vectordb.get()
        
        print(f"\n📊 Database Statistics:")
        print(f"Total documents: {len(data['ids'])}")
        print(f"Total embeddings: {len(data.get('embeddings', []))}")
        
        if data["ids"]:
            print(f"\n📋 Document IDs:")
            for i, doc_id in enumerate(data["ids"][:5]):  # Show first 5
                print(f"  {i+1}. {doc_id}")
            if len(data["ids"]) > 5:
                print(f"  ... and {len(data['ids']) - 5} more")
            
            print(f"\n📄 Sample Documents:")
            for i, doc in enumerate(data["documents"][:3]):  # Show first 3
                print(f"\n--- Document {i+1} ---")
                print(f"Content: {doc[:200]}{'...' if len(doc) > 200 else ''}")
                if data["metadatas"] and i < len(data["metadatas"]):
                    print(f"Metadata: {data['metadatas'][i]}")
            
            if len(data["documents"]) > 3:
                print(f"\n... and {len(data['documents']) - 3} more documents")
            
            print(f"\n🏷️  All Metadata:")
            sources = set()
            for meta in data.get("metadatas", []):
                if meta and "source" in meta:
                    sources.add(meta["source"])
            
            if sources:
                print(f"Sources found: {list(sources)}")
            else:
                print("No source metadata found")
                
        else:
            print("\n❌ No documents found in the database!")
            print("💡 The database exists but is empty. Try running /train endpoint.")
            
    except Exception as e:
        print(f"❌ Error accessing database: {str(e)}")
        print(f"💡 Database path: {db_path}")
        
    finally:
        # Cleanup
        try:
            del vectordb
        except:
            pass

def list_all_chromadb_locations():
    """Helper function to find all ChromaDB locations"""
    print("\n🔍 All ChromaDB Locations:")
    
    # Check current directory
    if os.path.exists("db"):
        print(f"✅ Found: ./db")
        
    # Check temp directory
    temp_dir = tempfile.gettempdir()
    chroma_dirs = glob.glob(os.path.join(temp_dir, "chroma_db_*"))
    for path in chroma_dirs:
        mod_time = os.path.getctime(path)
        from datetime import datetime
        time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"✅ Found: {path} (created: {time_str})")
    
    # Check saved path
    if os.path.exists(".db_path"):
        with open(".db_path", "r") as f:
            saved_path = f.read().strip()
            exists = "✅" if os.path.exists(saved_path) else "❌"
            print(f"{exists} Saved path: {saved_path}")

if __name__ == "__main__":
    print("🔎 Vector Database Debugger")
    print("=" * 50)
    
    # First, show all possible locations
    list_all_chromadb_locations()
    
    print("\n" + "=" * 50)
    
    # Then debug the active database
    debug_vectorstore()