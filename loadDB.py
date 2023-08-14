from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
# from helper_fns import HuggingFaceInstructEmbeddings

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

new_db = FAISS.load_local("sunbird_data", embeddings)
query = " what is sunbird ? and tell about some of its products"
docs = new_db.similarity_search(query)

print(docs[0])