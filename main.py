import asyncio
from llama_index.core.workflow import Context
from workflow import DITAdvisorWorkflow

async def chat_loop():
    print("DIT Advisor AI")
    
    workflow = DITAdvisorWorkflow(workflow_name="DIT Advisor", timeout=120.0)
    
    ctx = Context(workflow)
    
    while True:
        user_input = input("\nStudent: ")
        if user_input.lower() in ['exit', 'quit']: break
        if not user_input.strip(): continue
        
        handler = workflow.run(
            user_msg=user_input,
            ctx=ctx
        )

        async for event in handler.stream_events():
            if hasattr(event, "tool_call"):
                tc = event.tool_call
                print(f"\n  [AGENT ACTION]")
                print(f"\tTool   : {tc.tool_name}")
                print(f"\tParams : {tc.tool_kwargs}")
            
            elif hasattr(event, "tool_output"):
                tr = event.tool_output
                print(f"\n  [AGENT RESPONSE]\n")
                print(f"Tool Output: {str(tr.content)}")

        result = await handler
        print("-" * 40)
        print(f"Advisor: {result}")
        print("-" * 40)
        
        # print(f"\nAdvisor: {result}")

if __name__ == "__main__":
    asyncio.run(chat_loop())