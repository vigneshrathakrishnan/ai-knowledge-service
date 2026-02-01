# Comprehensive ChromaDB Debug Script

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import sqlite3
import json

load_dotenv()

def check_database_files():
    """Check what files exist in the db directory"""
    db_path = os.path.abspath("db")
    print(f"🔍 Checking database directory: {db_path}")
    
    if not os.path.exists(db_path):
        print("❌ Database directory does not exist!")
        return False
    
    print(f"✅ Database directory exists")
    
    # List all files
    print("\n📁 Files in database directory:")
    for root, dirs, files in os.walk(db_path):
        level = root.replace(db_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            print(f"{subindent}{file} ({size} bytes)")
    
    return True

def check_sqlite_database():
    """Check the SQLite database directly"""
    db_path = os.path.abspath("db")
    chroma_db_file = os.path.join(db_path, "chroma.sqlite3")
    
    if not os.path.exists(chroma_db_file):
        print("❌ chroma.sqlite3 file not found!")
        return
    
    print(f"\n🗄️  Checking SQLite database: {chroma_db_file}")
    
    try:
        conn = sqlite3.connect(chroma_db_file)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📊 Tables in database: {[table[0] for table in tables]}")
        
        # Check embeddings table
        if ('embeddings',) in tables:
            cursor.execute("SELECT COUNT(*) FROM embeddings;")
            count = cursor.fetchone()[0]
            print(f"📈 Number of embeddings: {count}")
            
            if count > 0:
                cursor.execute("SELECT id, document FROM embeddings LIMIT 3;")
                sample_docs = cursor.fetchall()
                print(f"\n📄 Sample documents:")
                for i, (doc_id, document) in enumerate(sample_docs, 1):
                    print(f"  {i}. ID: {doc_id}")
                    print(f"     Document: {document[:100] if document else 'None'}...")
        
        # Check collections table
        if ('collections',) in tables:
            cursor.execute("SELECT * FROM collections;")
            collections = cursor.fetchall()
            print(f"📚 Collections: {collections}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error reading SQLite database: {e}")

def test_chroma_connection():
    """Test ChromaDB connection and retrieval"""
    print(f"\n🔗 Testing ChromaDB connection...")
    
    try:
        embeddings = OpenAIEmbeddings()
        db_path = os.path.abspath("db")
        
        print(f"📁 Database path: {db_path}")
        print(f"🤖 Embeddings model: {embeddings}")
        
        # Try to connect
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
        print(f"✅ ChromaDB connection successful")
        
        # Get all data
        data = vectordb.get()
        
        print(f"📊 Retrieved data structure:")
        print(f"  - IDs: {len(data.get('ids', []))} items")
        print(f"  - Documents: {len(data.get('documents', []))} items")
        print(f"  - Metadatas: {len(data.get('metadatas', []))} items")
        print(f"  - Embeddings: {len(data.get('embeddings', []))} items")
        
        if data.get('ids'):
            print(f"\n📋 Sample IDs: {data['ids'][:3]}")
            
        if data.get('documents'):
            print(f"\n📄 Sample documents:")
            for i, doc in enumerate(data['documents'][:2]):
                print(f"  {i+1}. {doc[:150]}...")
                
        if data.get('metadatas'):
            print(f"\n🏷️  Sample metadata: {data['metadatas'][:2]}")
        
        # Try a similarity search
        if data.get('documents'):
            print(f"\n🔍 Testing similarity search...")
            results = vectordb.similarity_search("test query", k=1)
            print(f"✅ Similarity search returned {len(results)} results")
            if results:
                print(f"   First result: {results[0].page_content[:100]}...")
        
        del vectordb
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        import traceback
        print(f"📍 Full traceback:\n{traceback.format_exc()}")
        return False

def check_openai_connection():
    """Check if OpenAI embeddings are working"""
    print(f"\n🤖 Testing OpenAI embeddings...")
    
    try:
        embeddings = OpenAIEmbeddings()
        test_embed = embeddings.embed_query("test")
        print(f"✅ OpenAI embeddings working (dimension: {len(test_embed)})")
        return True
    except Exception as e:
        print(f"❌ OpenAI embeddings failed: {e}")
        return False

def main():
    print("🚀 ChromaDB Comprehensive Debug")
    print("=" * 50)
    
    # Step 1: Check files
    if not check_database_files():
        print("\n💡 Suggestion: Run the /train endpoint first to create the database")
        return
    
    # Step 2: Check SQLite directly
    check_sqlite_database()
    
    # Step 3: Check OpenAI connection
    check_openai_connection()
    
    # Step 4: Test ChromaDB connection
    test_chroma_connection()
    
    print("\n" + "=" * 50)
    print("🏁 Debug complete!")

if __name__ == "__main__":
    main()