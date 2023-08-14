import os
import PyPDF2
from helper_fns import *
from tqdm import tqdm 

import os
import PyPDF2
from tqdm import tqdm  # For progress bar


# Replace 'path_to_folder' with the actual path to your folder containing PDFs
folder_path = 'pdfs'

# List all files in the folder
pdf_files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]

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
        print(chunk)


print("starting ingestion")

vector_Store = get_vectorstore(chunks)

vector_Store.save_local("sunbird_data")

print("vector store created")