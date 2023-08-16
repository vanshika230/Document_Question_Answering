from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS, Milvus
import spacy


def get_pdf_text(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def coreferencing(text):
    nlp = spacy.load("en_coreference_web_trf")
    doc = nlp (text)
    offset = 0
    reindex = []
    for chain in doc.spans:
        for idx, span in enumerate(doc.spans[chain]):        
            if idx > 0:
                reindex.append([span.start_char, span.end_char, doc.spans[chain][0].text])

    for span in sorted(reindex, key=lambda x:x[0]):    
        text = text[0:span[0] + offset] + span[2] + text[span[1] + offset:]
        offset += len(span[2]) - (span[1] - span[0])    
    
    return text

def create_embeddings(text):
    model = INSTRUCTOR('hkunlp/instructor-base')
    instruction = "Represent the text:"
    embeddings = model.encode([[instruction , sentence]])
    return embeddings


# dont need the embeddings function if we use the vectorstore-method

# create text-chunks by preprocessing and coreferencing paras
# then send this text-chunk into vector store for further retrieval


def get_vectorstore(text_chunks): #chnages to be made here vectorstore ko milvus banana hai
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


