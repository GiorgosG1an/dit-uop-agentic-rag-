import asyncio
import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core.retrievers import VectorIndexAutoRetriever, VectorIndexRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from setup_qdrant import get_vector_store

import logging
import sys
from llama_index.core import set_global_handler

set_global_handler("simple")
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Describe the metadata schema for the LLM
vector_store_info = VectorStoreInfo(
    content_info="Detailed information about university courses, including syllabus, learning outcomes, and schedules.",
    metadata_info=[
        MetadataInfo(
            name="semester", 
            type="integer", 
            description="The semester the course is taught (e.g., 1 through 8)"
        ),
        MetadataInfo(
            name="ects", 
            type="integer", 
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
            description="List of professor last names teaching the course"
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
    temperature=0
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
    use_fp16=True,
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
    empty_query_is_text_all=True,
    vector_store_query_mode=VectorStoreQueryMode.HYBRID,
)

query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    node_postprocessors=[reranker],
    llm=llm,
)

semantic_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=15,
    vector_store_query_mode=VectorStoreQueryMode.HYBRID,
    alpha=0.7,
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
    node_postprocessors=[reranker],
    llm=llm,
    text_qa_template=qa_prompt_tmpl,
)

semantic_search_tool = QueryEngineTool(
    query_engine=semantic_query_engine,
    metadata=ToolMetadata(
        name="semantic_course_search",
        description=(
            "Use this tool to search for course concepts, syllabus details, learning outcomes, "
            "or what a student will learn in a specific field. "
            "Best for open-ended or conceptual questions like 'What will I learn in Machine Learning?'"
        )
    )
)

filtered_search_tool = QueryEngineTool(
    query_engine=query_engine, 
    metadata=ToolMetadata(
        name="filtered_course_search",
        description=(
            "Use this tool ONLY when the user explicitly asks to filter by "
            "semester, ECTS, difficulty, season, or prerequisites. "
            "Do NOT use this for general questions about course content."
        )
    )
)

advisor_system_prompt = """
Είστε ο επίσημος Ψηφιακός Ακαδημαϊκός Σύμβουλος (AI) του Τμήματος Πληροφορικής και Τηλεπικοινωνιών του Πανεπιστημίου Πελοποννήσου.
Ο ρόλος σας είναι να καθοδηγείτε τους φοιτητές στην ακαδημαϊκή τους πορεία.

ΚΑΝΟΝΕΣ ΜΟΡΦΟΠΟΙΗΣΗΣ:
- Χρησιμοποιήστε **Επικεφαλίδες (##)** για να διαχωρίζετε τις ενότητες της απάντησης.
- Χρησιμοποιήστε **Bold (έντονη γραφή)** για σημαντικούς όρους, κωδικούς μαθημάτων και ονόματα τεχνολογιών.
- Χρησιμοποιήστε **Bullet points (-)** για την ανάλυση της ύλης και των αλγορίθμων.
- Όταν συγκρίνετε μαθήματα ή παρουσιάζετε προαπαιτούμενα και ECTS, χρησιμοποιήστε **Πίνακες (Tables)** για μέγιστη σαφήνεια.
- Διαχωρίστε τις θεματικές ενότητες με οριζόντιες γραμμές (---).

ΟΔΗΓΙΕΣ ΠΕΡΙΕΧΟΜΕΝΟΥ:
- Να είστε επαγγελματίας, ενθαρρυντικός και αυστηρά αντικειμενικός.
- Εάν δεν γνωρίζετε την απάντηση, παραδεχτείτε το και προτείνετε τον επίσημο οδηγό σπουδών (PDF).
- Αναλύστε διεξοδικά τεχνολογίες και έννοιες (όχι μόνο keywords).
- Αναφέρετε πάντα τα προαπαιτούμενα για τεχνικές κατευθύνσεις.
- Απαντάτε πάντα στη γλώσσα του φοιτητή (Ελληνικά ή Αγγλικά).
"""
agent = FunctionAgent(
    name="DIT_Advisor",
    description="Official Academic Advisor AI for the Informatics Department.",
    system_prompt=advisor_system_prompt,
    tools=[semantic_search_tool, filtered_search_tool],
    llm=llm,
)

async def chat_loop():
    print("==================================================")
    print("🎓 DIT UoP Academic Advisor AI (Workflow Mode)")
    print("Type 'exit' or 'quit' to terminate the session.")
    print("==================================================")
    
    while True:
        try:
            user_input = input("\nStudent: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nAdvisor: Good luck with your studies! Goodbye.")
                break
                
            if not user_input.strip():
                continue
                
            print("\nThinking...")
            
            response = await agent.run(user_msg=user_input)
            
            print(f"\nAdvisor: {str(response)}")
            
        except KeyboardInterrupt:
            print("\n\nAdvisor: Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error]: An unexpected error occurred: {e}")

if __name__ == "__main__":
    
    asyncio.run(chat_loop())