"""tools used in the workflow"""
import asyncio
from typing import Dict
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata, AsyncBaseTool
from retrieval import semantic_query_engine, query_engine 

lock = asyncio.Lock()
async def semantic_search(input_query: str) -> str:
    """Performs a broad semantic search."""
    async with lock:
        response = await semantic_query_engine.aquery(input_query)
        return str(response)

semantic_search_tool = FunctionTool.from_defaults(
    async_fn=semantic_search,
    name="semantic_course_search",
    description=(
        "Use this tool for general, open-ended or conceptual questions about courses, "
        "such as 'What courses are related to Artificial Intelligence?', "
        "'Which courses cover Machine Learning topics?', "
        "'Give me an overview of AI-related courses in the department'. "
        "It searches broadly across all course materials and returns relevant results. "
        "ALWAYS use this as the primary tool for broad topic questions."
    )
)

filtered_search_tool = QueryEngineTool(
    query_engine=query_engine, # This uses VectorIndexAutoRetriever
    metadata=ToolMetadata(
        name="filtered_course_search",
        description=(
            "MANDATORY TOOL for any query containing specific constraints. "
            "Use this tool ONLY when the user asks to filter courses by exact parameters: "
            "semester (e.g., 6th), ECTS credits (e.g., 5), difficulty (e.g., 'Advanced'), "
            "season (e.g., 'Εαρινό'), or specific prerequisites. "
            "Do NOT use this for general conceptual questions."
            "Additionally useful when you need to narrow down results by specific criteria AFTER an initial broad search."
        )
    )
)

ADVISOR_TOOLS: Dict[str, FunctionTool | QueryEngineTool] = {
    "semantic_course_search": semantic_search_tool,
    "filtered_course_search": filtered_search_tool
}