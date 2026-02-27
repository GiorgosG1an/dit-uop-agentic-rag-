from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client import models

from llama_index.vector_stores.qdrant import QdrantVectorStore

client = QdrantClient(
    host='localhost',
    port=6333,
)

aclient = AsyncQdrantClient(
    host='localhost',
    port=6333,
)

COLLECTION_NAME = "dit_uop_agentic_rag"
ΒΑTCH_SIZE = 2

vector_store = QdrantVectorStore(
    client=client,
    aclient=aclient,
    collection_name=COLLECTION_NAME,
    enable_hybrid=True,
    fastembed_sparse_model="Qdrant/bm25",
    dense_vector_name="text-dense",
    sparse_vector_name="text-sparse",
    batch_size=ΒΑTCH_SIZE,
)

if not client.collection_exists(COLLECTION_NAME):
    print(f"Creating collection {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"text-dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
        sparse_vectors_config={"text-sparse": models.SparseVectorParams()}
    )
    print(f"Collection {COLLECTION_NAME} created")
else:
    print(f"Collection {COLLECTION_NAME} already exists")


# Numeric Indexes
client.create_payload_index(COLLECTION_NAME, "semester", field_schema=models.PayloadSchemaType.INTEGER)
client.create_payload_index(COLLECTION_NAME, "ects", field_schema=models.PayloadSchemaType.INTEGER)

# Categorical Indexes
client.create_payload_index(COLLECTION_NAME, "difficulty", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(COLLECTION_NAME, "season", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(COLLECTION_NAME, "course_id", field_schema=models.PayloadSchemaType.KEYWORD)

# Array/List Indexes
client.create_payload_index(COLLECTION_NAME, "instructors", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(COLLECTION_NAME, "career_paths", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(COLLECTION_NAME, "prerequisites", field_schema=models.PayloadSchemaType.KEYWORD)
client.create_payload_index(COLLECTION_NAME, "keywords", field_schema=models.PayloadSchemaType.KEYWORD)

def get_vector_store() -> QdrantVectorStore:
    return vector_store
        