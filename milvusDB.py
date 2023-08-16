import os
from helper_fns import *
from tqdm import tqdm 



# Replace 'path_to_folder' with the actual path to your folder containing PDFs
folder_path = 'pdfs'

# List all files in the folder
pdf_files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]

all_chunks = []  # Define the list to accumulate all chunks

for pdf_file in pdf_files:
    pdf_path = os.path.join(folder_path, pdf_file)
    
    # Get text content from the PDF
    content = get_pdf_text(pdf_path)
    content = preprocess_text(content)
    
    # Split text into chunks
    chunks = get_text_chunks(content)
    
    # Process each chunk and apply coreferencing
    for chunk in tqdm(chunks, desc=f"Processing {pdf_file}"):
        chunk = coreferencing(chunk)
        all_chunks.append(chunk)  # Append the processed chunk to the list

print("starting ingestion")

vector_db = Milvus.from_texts(
    all_chunks,
    embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base"),
    collection_name="SunbirdData",
    connection_args={"host": "127.0.0.1", "port": "19530"},
)

# vector_Store = get_vectorstore(chunks)

print("vector store created")

# load saved milvus collection
vector_db: Milvus = Milvus(
    embedding_function=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base"),
    collection_name="SunbirdData",
    connection_args={"host": "127.0.0.1", "port": "19530"},
)

query1 = "what is sunbird about?"

ans = vector_db.similarity_search(query = query1)

print("answer is : ", ans)
