#Sri Rama Jayam
#Running the FAISS_RAG_1.py
from FAISS_RAG_1 import FinancialAssistant
import os
from dotenv import load_dotenv 

load_dotenv('api_key.env')
openai_api_key = os.environ.get("OPENAI_API_KEY")
data_path = os.path.expanduser("~/Google-Vision-Hackathon/FIN_RAG/document-retrieval/myenv/dataset")
faiss_path = os.path.expanduser("~/Google-Vision-Hackathon/FIN_RAG/document-retrieval/myenv/document-retrieval-faiss/vector-store/financial_faiss")

assistant = FinancialAssistant(data_path, faiss_path)

query = "What is the difference between savings and investment"
response = assistant.generate_response(query)
print("\nQuery: ", query)
print("\nResponse: ", response)

