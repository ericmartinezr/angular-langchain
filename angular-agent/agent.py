from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph_supervisor import create_supervisor
from agents.code_generator import code_generator_agent
from agents.project_generator import project_generator_agent
from utils.logger import logger
from dotenv import load_dotenv

load_dotenv()

supervisor_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    # max_tokens=2048,
    timeout=None,
    max_retries=2
)

checkpointer = InMemorySaver()
supervisor = create_supervisor(
    model=supervisor_model,
    agents=[project_generator_agent, code_generator_agent],
    prompt=(
        """
        You're an expert manager agent. Your job is to manage two agents that will generate angular code.
        **INSTRUCTIONS:**
        - `coder_generator_agent`: The code generator agent. Assign code generation tasks to this agent.
        - `project_generator_agent`: The project generator agent. Assign project structuring and project structure validation to this agent.
        * Assign tasks to these agents sequentially, do not call them in parallel.
        * **DO NOT** do any work yourself.
        * **DO NOT** use your own angular knowledge. Rely on the agents to do the work.
        """.strip()
    ),
    output_mode="last_message",
).compile(checkpointer=checkpointer)


message = {
    "messages": [
        {
            "role": "user",
            "content": """
            Generate an angular v20 project. It must contain:
            - The project structure with the configuration to run the project.
            **CRITICAL:** you must use the latest angular version available.
            """.strip()
        },
        {
            "role": "user",
            "content": """
            To the current project you generated you have to:
            - Add two components: `calculator` and `calculator-result`.
            - Communicate both components with inputs and outputs.
            """.strip()
        },
        {
            "role": "user",
            "content": """
            You have to validate that the project was correctly created.
            """.strip()
        },
        {
            "role": "user",
            "content": """
            You have to let me know the path where the project was created.
            """.strip()
        }
    ]
}


class CustomHandler(BaseCallbackHandler):
    def on_tool_start(self,
                      serialized: dict[str, Any],
                      input_str: str, **kwargs: Any):
        logger.debug("Tool init")
        logger.debug(f"Serialized: {serialized}")
        logger.debug(f"Input str: {input_str}")

    def on_tool_end(self, output: Any, **kwargs: Any):
        logger.debug("Tool end")
        logger.debug(f"Output: {output}")


for chunk in supervisor.stream(message, {"configurable": {"thread_id": "1"}, "recursion_limit": 50, "callbacks": [CustomHandler()]}):
    logger.debug(f"{chunk}")
    if "supervisor" in chunk:
        supervisor_messages = chunk.get('supervisor', {}).get('messages', [])

        for message in supervisor_messages:
            logger.debug("Message")
            logger.debug("---" * 50)
            logger.debug(message.content)
