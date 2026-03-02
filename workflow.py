import asyncio
from typing import List, Literal
from pydantic import BaseModel, Field
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step, Context
from llama_index.core.memory import Memory
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.google_genai import GoogleGenAI
import os

from tools import ADVISOR_TOOLS
from prompts import ADVISOR_SYSTEM_PROMPT

# --- Pydantic Router Schema ---
class QueryPlan(BaseModel):
    intent: Literal["direct_chat", "broad_search", "specific_filter", "complex_multi_step"] = Field(
        ..., 
        description=(
            "Η στρατηγική ανάκτησης δεδομένων. Επιλογή βάσει των εξής κανόνων:\n"
            "1. 'direct_chat': Χαιρετισμοί, γενικές ερωτήσεις εκτός ακαδημαϊκού πλαισίου ή απαντήσεις που βασίζονται ΑΠΟΚΛΕΙΣΤΙΚΑ στο ιστορικό της συνομιλίας χωρίς ανάγκη νέων δεδομένων. ΟΧΙ για ερωτήσεις σχετικές με το τμήμα.\n"
            "2. 'specific_filter': Χρήση ΜΟΝΟ όταν η ερώτηση περιέχει συγκεκριμένα μεταδεδομένα όπως: εξάμηνο (1-8), ECTS, όνομα καθηγητή, ή κωδικό μαθήματος.\n"
            "3. 'complex_multi_step': Χρήση ΠΑΝΤΑ όταν ο χρήστης αναφέρει 2 ή περισσότερα μαθήματα, συγκρίνει μαθήματα, ή ζητά λεπτομέρειες για μια λίστα μαθημάτων που βρέθηκαν προηγουμένως.\n"
            "4. 'broad_search': Για εννοιολογικές ερωτήσεις (π.χ. 'τι είναι η επιστήμη υπολογιστών'), γενικές πληροφορίες για το τμήμα (π.χ Λίγα λόγια για το τμήμα) ή όταν η ερώτηση δεν ταιριάζει στα παραπάνω."
        )
    )
    
    expanded_query: str = Field(
        ..., 
        description=(
            "ΜΟΝΟ εάν η αρχική ερώτηση χρειάζεται εμπλουτισμό, αλλίως παραμένει ίδια."
            "Η βελτιστοποιημένη ερώτηση για το vector store. "
            "Πρέπει να είναι στα Ελληνικά. Αφαίρεσε φράσεις όπως 'θα ήθελα να μάθω', 'πες μου για'. "
            "Εστίασε σε ουσιαστικά και τεχνικούς όρους."
        )
    )
    
    sub_queries: List[str] = Field(
        default_factory=list,
        description=(
            "Υποχρεωτικό ΜΟΝΟ για το 'complex_multi_step'. "
            "Δημιούργησε μια λίστα που σπάνε την αρχική ερώτηση σε πιο εύκολα υπο-ερωτήματα."
            "Τα υπο-ερωτήματα πρέπει να είναι σημασιολογικά όμοια με την αρχική ερώτηση"
            "Απέφυγε όρους όπως 'Πανεπιστήμιο Πελοποννήσου', 'Τμήμα Πληροφορικής και Τηλεπικοινωνιών'."
            "Πρέπει να είναι φιλικά προς την ανάκτηση."
        )
    )
    
    reasoning: str = Field(
        ..., 
        description="Σύντομη αιτιολόγηση στα ελληνικά. Πρέπει να εξηγεί γιατί επιλέχθηκε το συγκεκριμένο intent βάσει των λέξεων-κλειδιών του χρήστη, σαν εσωτερικός μονόλογος."
    )

# --- Cleaned Custom Events ---
class RouterEvent(Event):
    plan: QueryPlan

class ContextGatheredEvent(Event):
    context: str

# --- Workflow Definition ---
class DITAdvisorWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.1)
        self.system_prompt = ADVISOR_SYSTEM_PROMPT

        self.memory = Memory.from_defaults(
            token_limit=10000,
            chat_history=[ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt)]
        )

    @step
    async def route_query(self, ctx: Context, ev: StartEvent) -> RouterEvent | ContextGatheredEvent:
        
        user_msg = ev.get("user_msg")
        # await ctx.store.set("original_query", user_msg)
        self.memory.put(ChatMessage(role="user", content=user_msg))

        chat_history = self.memory.get()
        
        structured_llm = self.llm.as_structured_llm(output_cls=QueryPlan)
        response = await structured_llm.achat(chat_history)
        plan: QueryPlan = response.raw
        
        if plan.intent == "direct_chat":
            return ContextGatheredEvent(context="No academic context needed.")
            
        return RouterEvent(plan=plan)

    @step
    async def execute_retrieval(self, ctx: Context, ev: RouterEvent) -> ContextGatheredEvent:
        plan = ev.plan
        accumulated_context = ""
        # original_query = await ctx.store.get("original_query")

        # query = f"{original_query} {plan.expanded_query}".strip()
        if plan.sub_queries and plan.intent == "complex_multi_step":
            tasks = [ADVISOR_TOOLS["semantic_course_search"].acall(query) for query in plan.sub_queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, res in enumerate(results):
                accumulated_context += f"\n--- Results for: {plan.sub_queries[i]} ---\n{res}\n"

        elif plan.intent == "specific_filter":
            
            res = await ADVISOR_TOOLS["filtered_course_search"].acall(plan.expanded_query)
            accumulated_context = str(res)
        else:
            res = await ADVISOR_TOOLS["semantic_course_search"].acall(plan.expanded_query)
            accumulated_context = str(res)
        
        return ContextGatheredEvent(context=accumulated_context)

    @step
    async def synthesize(self, ctx: Context, ev: ContextGatheredEvent) -> StopEvent:
        
        context_info = ChatMessage(
            role=MessageRole.SYSTEM, 
            content=f"RETRIEVED ACADEMIC CONTEXT:\n{ev.context}"
        )
        
        current_history = self.memory.get() + [context_info]

        response = await self.llm.achat(current_history)
        
        self.memory.put(response.message)
        
        return StopEvent(result=str(response.message.content))