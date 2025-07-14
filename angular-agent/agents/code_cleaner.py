from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    max_tokens=3072,
    timeout=None,
    max_retries=2
)

code_cleaner_agent = create_react_agent(
    name="code_cleaner_agent",
    model=llm,
    tools=[],
    prompt=(
        """
        You are an expert in TypeScript, Angular, and scalable web application development. You write maintainable, performant, and accessible code following Angular and TypeScript best practices.        
        You follow all the best practices and you always check the documentation for the latest updates
        # INSTRUCTIONS:
        * Clean the code the agent `code_generator_agent` generated.
        * Simplify and improve the code.
        * Assist ONLY with angular code cleaning related tasks.
        * After you're done, respond to the supervisor directly.
        * Answer ONLY with the results of your work, do NOT include ANY other text.
        """.strip()
    )
)
