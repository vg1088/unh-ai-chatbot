""" Script to load pdf into qdrant vector database, needs pdf name as command line argument """
import os
import sys
import PyPDF2
from  qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from langchain.text_splitter import CharacterTextSplitter

CHUNK_SIZE = 300

def get_docs(pdf_path):
    """ Converts pdf into string of text"""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() or ""
    return text

def path_from_name(file_name):
    """Converts name to file path """
    exec_path = os.path.dirname(__file__)
    file_path = os.path.join(exec_path, file_name)

    return file_path

path = path_from_name(sys.argv[1])
collect = sys.argv[2]

docs = get_docs(path)

text_splitter = CharacterTextSplitter( separator = "\n", chunk_size = CHUNK_SIZE,
                                      chunk_overlap = 100, length_function=len)

chunks = text_splitter.split_text(docs)

client = QdrantClient( host='localhost' )
embed_model = TextEmbedding()

embeds = embed_model.embed(chunks)

pl_text = []
for index, section in enumerate(chunks):
    payload = {str(index) : section}
    pl_text.append(payload)

embeds = models.Batch( ids=range(0, len(chunks)), vectors = list(embeds), payloads = pl_text)

client.upsert(collection_name = collect,
              points = embeds
             )