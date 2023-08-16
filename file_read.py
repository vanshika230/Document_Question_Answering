import os
from helper_fns import *
from tqdm import tqdm 


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


# check functioning of preprocessing function with print statements
# modified and used in milvusDB.py
