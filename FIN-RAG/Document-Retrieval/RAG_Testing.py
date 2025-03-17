#Sri Rama Jayam
#Running the FAISS_RAG_1.py
from RAG_Model import FinancialAssistant
import os
from dotenv import load_dotenv 

#Load the environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY") #Ensure the .env file is in the directory in which you are running the code
data_path = os.getenv("DATA_PATH")  # Load data path from environment variable
faiss_path = os.getenv("FAISS_PATH")  # Load FAISS path from environment variable

# Check if paths are set
if not data_path or not faiss_path:
    raise ValueError("DATA_PATH and FAISS_PATH must be set in environment variables.")

# Initialize the Financial Assistant
assistant = FinancialAssistant(data_path, faiss_path)

# Sample Query - Can be replaced with any query
query = "What is the difference between savings and investment"
response = assistant.generate_response(query)

# Print results
print("\nQuery: ", query)
print("\nResponse: ", response)


