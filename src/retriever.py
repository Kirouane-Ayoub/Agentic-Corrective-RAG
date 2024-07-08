import settings as s
from llama_index.core import (
    ServiceContext,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from models import embed_model, llm

# set up the global settings

Settings.embed_model = embed_model
Settings.llm = llm

# Create the service context with the cohere model for generation and embedding model
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
# load data

docs = SimpleDirectoryReader(s.DATA_SOURCE).load_data()

# build index
index = VectorStoreIndex.from_documents(docs, service_context=service_context)
retriever = index.as_retriever()
