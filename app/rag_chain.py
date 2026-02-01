# app/rag_chain.py (MINIMAL UPDATES)

import os
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.chat_models import ChatOpenAI  # ✅ Fixed import
from langchain.prompts import PromptTemplate
from .db import load_vectorstore

def detect_query_type(question: str) -> str:
    """Enhanced intent detection using LLM reasoning"""
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=10)
    
    intent_prompt = PromptTemplate(
        template="""Classify this question into one category:
        Question: "{question}"

        Categories:
        - summary: asking for summary, main points, or overall content
        - overview: asking what something is about, general explanation  
        - general: specific questions, facts, details

        Response (one word only):""",
                input_variables=["question"]
    )
    
    try:
        chain = LLMChain(llm=llm, prompt=intent_prompt)
        intent = chain.run(question=question).strip().lower()
        
        if intent in ["summary", "overview", "general"]:
            return intent
        return "general"  # fallback
    except:
        # Fallback to simple keyword detection
        q = question.lower()
        if "summary" in q or "summarize" in q:
            return "summary"
        elif "overview" in q or "what is this about" in q or "main theme" in q:
            return "overview"
        return "general"

def get_rag_chain(question: str):
    """Handle different types of queries with smart not-found detection"""
    try:
        vectordb = load_vectorstore()
        if vectordb is None:
            return {"error": "No knowledge base found. Please train the model first using /train endpoint."}
        
        query_type = detect_query_type(question)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})  # ✅ Reduced chunks
        llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-16k")  # ✅ 16k model

        if query_type == "summary":
            docs = vectordb.get(include=["documents"])
            if not docs["documents"]:
                return "No documents found in knowledge base to summarize."
            
            # ✅ Limit text size to prevent token overflow
            text = " ".join(docs["documents"])
            if len(text) > 8000:  # ~2000 tokens
                text = text[:8000] + "..."
            
            prompt = PromptTemplate(
                template="""Provide a comprehensive summary of the following content. If the content doesn't contain enough meaningful information, respond with "I don't have sufficient content in the knowledge base to provide a meaningful summary."

            Content:
            {text}

            Summary:""",
                input_variables=["text"]
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run(text=text)

        elif query_type == "overview":
            docs = retriever.get_relevant_documents(question)
            if not docs:
                return "I don't have relevant information in the knowledge base to provide an overview of this topic."
            
            text = " ".join([d.page_content for d in docs])
            if len(text) > 6000:  # Limit context
                text = text[:6000] + "..."
                
            prompt = PromptTemplate(
                template="""Provide an overview of the following content related to the question. If the content doesn't contain relevant information, respond with "I don't have enough relevant information in the knowledge base to provide an overview of this topic."

            Question: {question}
            Content: {text}

            Overview:""",
                input_variables=["text", "question"]
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run(text=text, question=question)

        else:  # General Q&A
            # ✅ Enhanced general QA with not-found detection
            docs = retriever.get_relevant_documents(question)
            if not docs:
                return "I don't have any relevant information in the knowledge base to answer your question."
            
            # Custom prompt for better not-found handling
            context = "\n".join([doc.page_content for doc in docs])
            if len(context) > 5000:
                context = context[:5000] + "..."
            
            qa_prompt = PromptTemplate(
                template="""Answer the question based on the following context. If the context doesn't contain information to answer the question, respond exactly with: "I don't have enough relevant information in the knowledge base to answer this question."

                Context:
                {context}

                Question: {question}

                Answer:""",
                input_variables=["context", "question"]
            )
            
            chain = LLMChain(llm=llm, prompt=qa_prompt)
            return chain.run(context=context, question=question)
            
    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}