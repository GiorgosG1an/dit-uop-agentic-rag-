import os
import glob
import frontmatter
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from setup_qdrant import get_vector_store

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    device='cuda',
)

docs_path = "./courses" 
documents = []

for file_path in glob.glob(os.path.join(docs_path, "*.md")):
    post = frontmatter.load(file_path)
    metadata = dict(post.metadata)
    
    document = Document(
        text=post.content,
        metadata={
            **metadata,
            "full_content": post.content
        },
        metadata_separator="\n",
        metadata_template="{key}=>{value}",
        text_template="Metadata:\n{metadata_str}\n\nContent:\n{content}",
        excluded_llm_metadata_keys=["unlocked_concepts", "keywords", "skills_acquired", "course_code", "full_content"],
        excluded_embed_metadata_keys=["full_content"],
    )
    documents.append(document)

node_parser = MarkdownNodeParser()
nodes = node_parser.get_nodes_from_documents(documents)

vector_store = get_vector_store()

storage_context = StorageContext.from_defaults(vector_store=vector_store)

print(f"Ingesting {len(nodes)} nodes into Qdrant...")
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    show_progress=True,
)
print("Ingestion complete!")