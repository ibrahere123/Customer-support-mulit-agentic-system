import os
import sys

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# === Configuration ===
DATA_FILE   = "support-agent/app/products.csv"
INDEX_FILE  = "product_index.faiss"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 5

# === Load Gemini API key from env var ===
ENV_VAR = "GEMINI_API_KEY"
if ENV_VAR not in os.environ:
    print(f"Error: environment variable {ENV_VAR} is not set", file=sys.stderr)
    sys.exit(1)
GEMINI_API_KEY = os.environ[ENV_VAR]

# === Load product metadata ===
if not os.path.exists(DATA_FILE):
    print(f"Error: data file not found at {DATA_FILE}", file=sys.stderr)
    sys.exit(1)
df = pd.read_csv(DATA_FILE)

# === Load FAISS index ===
if not os.path.exists(INDEX_FILE):
    print(f"Error: index file not found at {INDEX_FILE}", file=sys.stderr)
    sys.exit(1)
try:
    index = faiss.read_index(INDEX_FILE)
except Exception as e:
    print(f"Error loading FAISS index: {e}", file=sys.stderr)
    sys.exit(1)

# === Load sentence embedder ===
device   = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBED_MODEL, device=device)

# === Search function ===
def search_products(query: str, k: int = TOP_K) -> str:
    q_emb = embedder.encode([query]).astype("float32")
    D, I = index.search(q_emb, k)
    out = []
    for idx in I[0]:
        prod = df.iloc[idx]
        out.append(
            f"Brand:   {prod['Brand']}\n"
            f"Product: {prod['Product Name']}\n"
            f"Rating:  {prod['Ratings']}\n"
            f"Price:   Rs. {prod['Price']}\n"
            f"Link:    {prod['Product Link']}\n"
            + "-"*40
        )
    return "\n".join(out)

# === Build tools list ===
tools = [
    Tool(
        name="ProductSearchTool",
        func=search_products,
        description="Looks up top matching products for a user query",
    )
]

# === Prompt engineering ===
prefix = """
You are ShopBuddy, an AI assistant specialized in recommending products from a large catalog.
When the user asks for product recommendations, you must call the ProductSearchTool exactly like this:

  Action: ProductSearchTool
  Action Input: <search query>

After you receive the tool’s output, summarize the results back to the user in friendly language.
If the user’s request is unrelated to products, answer directly without using the tool.
"""

suffix = """
User: {input}

{format_instructions}

Thought:"""

# LangChain will fill in {format_instructions} for you
prompt = PromptTemplate(
    template=prefix + "\n" + suffix,
    input_variables=["input", "format_instructions"],
)

# === Instantiate the LLM ===
llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model="gemini-2.0-flash-lite",
    temperature=0.5,
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === Initialize the agent with custom prompt ===
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose= True,
    agent_kwargs={
        "prompt": prompt,               # our PromptTemplate
        # Note: ZERO_SHOT_REACT_DESCRIPTION already expects prefix/suffix via `prompt`
    },
)

# === Run loop ===
if __name__ == "__main__":
    print("🔎 Gemini Product Search Agent is ready!  (type ‘exit’ to quit)\n")
    while True:
        q = input("🗣️  You: ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            print("👋 Bye!")
            break
        try:
            resp = agent.run(q)
        except Exception as e:
            print(f"⚠️  Error: {e}", file=sys.stderr)
            continue
        print("🤖 Agent:", resp, "\n")
