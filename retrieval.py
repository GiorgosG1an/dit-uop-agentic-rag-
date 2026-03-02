import asyncio
import os
from dotenv import load_dotenv
from typing import List, Optional

from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core.retrievers import VectorIndexAutoRetriever, VectorIndexRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle


from setup_qdrant import get_vector_store

import logging
import sys
from llama_index.core import set_global_handler

set_global_handler("simple")
# logging.basicConfig(
#     stream=sys.stdout, 
#     level=logging.INFO, 
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logging.getLogger("httpx").setLevel(logging.WARNING)


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class FullCourseContextPostProcessor(BaseNodePostprocessor):
    """
    Small-to-Big Retrieval PostProcessor.

    Replaces retrieved small chunks with the full course content stored in metadata,
    and deduplicates so the LLM doesn't receive the same course multiple times.
    """
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        unique_doc_ids = set()
        new_nodes = []
        
        for node_with_score in nodes:
            # ref_doc_id is the unique ID of the parent Document
            doc_id = node_with_score.node.ref_doc_id
            
            if doc_id not in unique_doc_ids:
                unique_doc_ids.add(doc_id)
                
                # Fetch the full content we stored during ingestion
                full_content = node_with_score.node.metadata.get("full_content")
                
                if full_content:
                    # Create a copy to avoid mutating cached nodes
                    node_copy = node_with_score.node.model_copy()
                    node_copy.text = full_content 
                    new_nodes.append(NodeWithScore(node=node_copy, score=node_with_score.score))
                else:
                    new_nodes.append(node_with_score)
                    
        return new_nodes
    
full_course_postprocessor = FullCourseContextPostProcessor()

# Describe the metadata schema for the LLM
vector_store_info = VectorStoreInfo(
    content_info="Detailed information about university courses, including syllabus, learning outcomes, and schedules.",
    metadata_info=[
        MetadataInfo(
            name="semester", 
            type="string", 
            description="The semester the course is taught (e.g., 1 through 8)."
        ),
        MetadataInfo(
            name="ects", 
            type="string", 
            description="European Credit Transfer System points (e.g., 5 or 6)"
        ),
        MetadataInfo(
            name="difficulty", 
            type="string", 
            description="Difficulty level of the course (e.g., 'Beginner', 'Intermediate', 'Advanced')"
        ),
        MetadataInfo(
            name="season", 
            type="string", 
            description="The season the course is offered, strictly either 'Εαρινό' (Spring) or 'Χειμερινό' (Winter)"
        ),
        MetadataInfo(
            name="course_id", 
            type="string", 
            description="The unique identifier code for the course (e.g., 'αντ-προ')"
        ),
        MetadataInfo(
            name="instructors", 
            type="list[string]", 
            description="List of professor last names teaching the course, it may be blank"
        ),
        MetadataInfo(
            name="career_paths", 
            type="list[string]", 
            description="Target professions this course prepares students for (e.g., 'Software Developer')"
        ),
        MetadataInfo(
            name="prerequisites", 
            type="list[string]", 
            description="Course codes that must be completed before taking this course"
        )
    ]
)

llm = GoogleGenAI(
    model="gemini-2.0-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.1
)
Settings.llm = llm

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    device='cuda',
)
Settings.embed_model = embed_model

reranker = FlagEmbeddingReranker(
    top_n=5,
    model="BAAI/bge-reranker-v2-m3",
    use_fp16=False, # this set at True caused problem `expected scalar type Float but found Half`
)

vector_store = get_vector_store()
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    show_progress=True,
)
# Create the Auto-Retriever
retriever = VectorIndexAutoRetriever(
    index=index,
    vector_store_info=vector_store_info,
    llm=llm,
    similarity_top_k=15,
    vector_store_query_mode=VectorStoreQueryMode.HYBRID,
)

query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    node_postprocessors=[full_course_postprocessor], # removed reranker
    llm=llm,
)

semantic_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=15,
    vector_store_query_mode=VectorStoreQueryMode.HYBRID,
    alpha=0.6,
)

qa_prompt_tmpl_str = (
    "Οι πληροφορίες πλαισίου βρίσκονται παρακάτω.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Δεδομένων των πληροφοριών πλαισίου και χωρίς να χρησιμοποιήσεις εξωτερικές γνώσεις, απάντησε στο ερώτημα στα Ελληνικά.\n"
    "ΣΗΜΑΝΤΙΚΟ: Διατήρησε την τεχνική ορολογία της πληροφορικής (π.χ. cross validation, Naive Bayes, k-means, support vector machines) στα Αγγλικά. Μην τα μεταφράζεις.\n"
    "Δώσε την απάντησή σου με μορφή λίστας (bullet points).\n"
    "Ερώτημα: {query_str}\n"
    "Απάντηση: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
semantic_query_engine = RetrieverQueryEngine.from_args(
    retriever=semantic_retriever,
    node_postprocessors=[reranker, full_course_postprocessor],
    llm=llm,
    text_qa_template=qa_prompt_tmpl,
)