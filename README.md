# Google-Solution-Challenge<br />
Hackathon Organized by Hack2skill. <br />
Working on the Problem Statement of GenAI-Powered Financial Assistant for Better Investing Decisions . <br />
Participants: <br />
Srihari Prasad - 2nd Year Undergrad Student, Department of Mechanical Engineering, IIT Madras, Chennai<br />
Kaeshav Siddarthan RM- 2nd Year Undergrad Student, Department of Naval Architecture and Ocean Engineering, IIT Madras, Chennai<br />
Arjun A A R - 2nd Year Undergrad Student, Department of Naval Architecture and Ocean Engineering, IIT Madras, Chennai<br />
Lavlin Jaison - 2nd Year Undergrad Student, Department of Mechanical Engineering, IIT Madras, Chennai. <br />

The RAG pipeline for the code written in FIN_RAG/Document-Retrieval is as follows: <br />
![rag_pipeline](https://github.com/user-attachments/assets/32f1ac68-1305-48b5-af76-38797886d374)

The FIN_RAG/Document-Retrieval folder contains two .py files <br />
1. <ol> <li> <strong>RAG_Model.py</strong> builds the RAG model by loading the PDF files and creating chunks of the data from the files and creating a vector embedding in FAISS. Then the user query is also broken into chunks and embedded. Then a similarity search is performed based on the vector embeddings and the top 3 responses related to the query are retrieved from FAISS_Index. The retrieved documents are passed to gpt-4o-mini along with a structured system prompt to generate a detailed and organized response. <br />
2. <ol> <li> <strong>RAG_Testing.py</strong> The FinancialAssistant is called by importing the RAG_Model.py. The data path and the path(for the directory) for the creating of the FAISS vector index is entered and the FinancialAssistant is called. The generate_response function from the FinancialAssistant is called and the response to the query is printed. <br />
