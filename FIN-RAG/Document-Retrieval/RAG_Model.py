#Sri Rama Jayam
#Imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
import numpy as np
import os
import openai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

load_dotenv('api_key.env')
openai_api_key = os.environ.get("OPENAI_API_KEY")

class FinancialAssistant:
    def __init__(self, data_path : str, faiss_path : str):
        self.data_path = data_path
        self.faiss_path = faiss_path
        self.embeddings = OpenAIEmbeddings()
        self.openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(os.path.dirname(faiss_path), exist_ok=True)

    def data_loader(self) -> List[Any]:
        try:
            loader=PyPDFDirectoryLoader(self.data_path)
            raw_documents = loader.load()
            print(f"Loaded {len(raw_documents)} documents from {self.data_path}")
            return raw_documents
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            return []

    def text_splitter(self) -> List[Any]:
        raw_documents = self.data_loader()
        if not raw_documents:
            return []

        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 30000, chunk_overlap = 5000)
            docs = text_splitter.split_documents(raw_documents)
            print(f"Split documents into {len(docs)} chunks")
            return docs
        except Exception as e:
            print(f"Error handling text: {str(e)}")
            return []

    def create_vectorstore(self, force_recreate: bool = False) -> Optional[FAISS]:
        if os.path.exists(self.faiss_path) and not force_recreate:
            try:
                print(f"Loading existing vector store from {self.faiss_path}")
                return FAISS.load_local(
                    self.faiss_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading vector store {str(e)}")
                print("Will create a new vector store")

        docs = self.text_splitter()
        if not docs:
            return None

        try:
            print("creating a new vector store")
            vector_store = FAISS.from_documents(docs, self.embeddings)  
            vector_store.save_local(self.faiss_path)
            print(f"Vector store saved to {self.faiss_path}")
            return vector_store
        except Exception as e:
            print(f"Error creating vector store {str(e)}")
            return None

    def retrieve_documents(self, user_query: str, top_k : 3) -> List[Any]:
        vector_store = self.create_vectorstore()
        if not vector_store:
            return []

        try:
            retriever = vector_store.as_retriever(
                    search_type = "similarity",
                    search_kwargs={"k": top_k}
                    )
            retrieved_docs = retriever.invoke(user_query)
            print(f"Retrieved {len(retrieved_docs)} documents for {str(user_query)}")
            return retrieved_docs
        except Exception as e:
            print(f"Error retrieving documents {str(e)}")
            return []
    
    def generate_response(self, user_query: str, temperature: float = 0.7) -> str:
        retrieved_docs = self.retrieve_documents(user_query,top_k=3)
        if not retrieved_docs:
            return "I could not find relevant information to answer your query. Please try a different question."

        retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])
        try:
            system_prompt="""   
            You are a financial assistant specialized in financial investments in India.
            You only answer based on the knowledge provided.

            Whenever possible, present data in a structured **tabular format** using Markdown.
            Use the following table structure:

            | **Category**  | **Details**  |
            |--------------|------------|
            | **Key Insights** | Summary of findings |
            | **Financial Impact** | How it affects funding/investments |
            | **Government Role** | Policies, subsidies, or schemes involved |
            | **Challenges** | Risks & barriers to investment |
            | **Opportunities** | Growth areas & investment potential |

            If numerical data is available, format it **clearly in a table** with appropriate units.

            If a subjective answer is present in the retrieved data, explain it thoroughly in the following format:
            - **Introduction:** Briefly introduce the concept.
            - **Key Features:** List the important characteristics.
            - **Examples:** Provide real-world scenarios.
            - **Advantages & Disadvantages:** Compare pros and cons.
            - **Conclusion:** Summarize the main takeaway.

            Use **bullet points**, **numbered lists**, and **examples** to ensure clarity. Provide an in-depth, structured response rather than a short summary.

            If the answer is not fully in the data, say: 'Based on the available information, here’s what I can summarize.'

            --------------------
            The data:
            {retrieved_text}
            """
            response = self.openai_client.chat.completions.create(
                    model = "gpt-4o-mini",
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context:\n{retrieved_text}\n\nQuestion: {user_query}"}
                        ],
                        temperature = temperature,
                        max_tokens = 1000
                    )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating response {str(e)}")
            return "I encountered an error in generating a response. Please try again later."
