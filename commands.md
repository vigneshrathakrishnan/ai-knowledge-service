# source mangovenv/bin/activate
# uvicorn app.main:app --reload

<!-- if error -->
# pip uninstall -y langchain langchain-community langchain-openai
# pip install langchain==0.1.20 langchain-community==0.0.38 langchain-openai==0.0.8


# python -c "from app.db import create_vectorstore; create_vectorstore()"
# streamlit run ui/streamlit_app.py 


# python -c "from app.db import create_vectorstore; create_vectorstore(source='file', file_path='data/elon-tesla-quest-ashlee.pdf')"

# python -c "from app.db import create_vectorstore; create_vectorstore(source='url', url='https://navabrindsol.com/')"


# python -c "from app.db import debug_vectorstore; debug_vectorstore()"
