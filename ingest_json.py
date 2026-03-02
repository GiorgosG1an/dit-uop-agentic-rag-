import json
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from setup_qdrant import get_vector_store

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    device='cuda',
)

json_file_path = "website_data/dit_corpus.jsonl"
data = []
with open(json_file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

documents = []

for item in data:
    text_content = item.get("summary", "")

    metadata = {
        "url": item.get("url", ""),
        "title": item.get("title", ""),
        "category": item.get("category", ""),
        "language": item.get("language", ""),
        "keywords": item.get("keywords", []),
        "suggested_questions": item.get("suggested_questions", []),
        "last_modified": item.get("last_modified"),
        "full_content": text_content 
    }

    document = Document(
        text=text_content,
        metadata=metadata,
        metadata_separator="\n",
        metadata_template="{key}=>{value}",
        text_template="Metadata:\n{metadata_str}\n\nContent:\n{content}",
        excluded_llm_metadata_keys=["full_content", "keywords", "suggested_questions"],
        excluded_embed_metadata_keys=["full_content", "url", "last_modified", "category", "language"],
    )

    documents.append(document)

parser = SentenceSplitter(chunk_size=2048, chunk_overlap=100)
nodes = parser.get_nodes_from_documents(documents)


vector_store = get_vector_store()
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print(f"Ingesting {len(nodes)} nodes into Qdrant...")
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    show_progress=True,
)
print("Ingestion complete!")