import settings
from dotenv import load_dotenv
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere

load_dotenv()

llm = Cohere(model=settings.LLM_MODEL_NAME)


embed_model = CohereEmbedding(
    model_name=settings.EMBED_MODEL_NAME,
    input_type="search_query",
)
