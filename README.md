# MultiAgent Self-Learning Customer Support System

This repository contains a **multi-agent system** built with **Gemini LLM**, **LangChain**, and **LangGraph**, designed to handle product search, customer support, and self-learning tasks. The system is intended to be **generic, extensible, and production-ready**, capable of working with both **CSV files** and **databases**.

---

## Features

- **Product Agent**: Retrieves and filters product data from CSV or database sources.
- **Support Agent**: Handles customer queries using Gemini LLM.
- **Coordinator Agent**: Orchestrates workflows between agents using LangGraph.
- **Learning Agent** *(planned)*: Collects feedback and enhances system performance over time.
- **Vector Database Integration**: Uses Pinecone, Weaviate, or Qdrant for semantic search.
- **Data Cleaning**: Automatically handles messy price formats like `'Rs. 2005'` → `2005.0`.
- **Flexible Column Mapping**: Maps CSV or DB columns to product attributes.

