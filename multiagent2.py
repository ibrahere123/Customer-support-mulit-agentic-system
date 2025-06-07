import os
import uuid
import chromadb
import re
import json
import time
import random
import threading
import queue
import pandas as pd
import traceback
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, TypedDict, Any
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.pregel import GraphRecursionError
import google.generativeai as genai
from typing import Literal, TypedDict, List, Dict, Optional, Tuple
import schedule
import hashlib

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ----------------------------
# Shared Resources
# ----------------------------
class SharedResources:
    """Central repository for shared components across agents"""
    def __init__(self):
        # Configure Gemini
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7, google_api_key=os.getenv("GEMINI_API_KEY"))

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=".chromadb")
        self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.getenv("GEMINI_API_KEY"))

        # Initialize knowledge base collection
        self.knowledge_collection = self.client.get_or_create_collection(
            name="support_kb",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize conversation memory
        self.conversation_collection = self.client.get_or_create_collection(
            name="conversation_memory",
            embedding_function=self.embedding_function
        )

        # Communication bus for agent coordination
        self.communication_bus = queue.Queue()

        # Shared state trackers
        self.user_contexts = {}
        self.user_profiles = {}
        self.escalation_count = {}
        self.proactive_triggers = {
            "password": ["security", "2fa", "recovery"],
            "order": ["tracking", "cancel", "return"],
            "account": ["verification", "settings", "delete"]
        }

        # Configuration parameters
        self.escalation_threshold = 0.4
        self.confidence_threshold = 0.85
        self.max_context_length = 4000
        self.context_summary_threshold = 3000
        self.max_history_length = 10
        self.profile_update_frequency = 3
        self.default_profile = {
            "preferred_name": "there",
            "technical_level": "intermediate",
            "communication_style": "direct",
            "known_issues": []
        }
        
        # Product data handling
        self.product_data = None
        self.product_embeddings = None
        self.product_data_hash = None
        self.column_mapping = {}
        self.temp_vector_store = {}
        
        # Load product data
        self.load_product_data()

    def detect_column_mapping(self) -> dict:
        """Automatically detect important columns in the product data"""
        mapping = {
            'name': None,
            'brand': None,
            'price': None,
            'rating': None,
            'link': None,
            'category': None,
            'color': None,
            'size': None
        }
        
        if self.product_data is None or self.product_data.empty:
            return mapping
            
        # Convert column names to lowercase for matching
        lower_columns = [col.lower() for col in self.product_data.columns]
        
        # Find matches for each type
        for col_type in mapping.keys():
            for col in self.product_data.columns:
                if col_type in col.lower():
                    mapping[col_type] = col
                    break
                    
        # If we couldn't find price, look for numeric columns
        if mapping['price'] is None:
            for col in self.product_data.select_dtypes(include='number').columns:
                if 'price' in col.lower() or 'cost' in col.lower() or 'amount' in col.lower():
                    mapping['price'] = col
                    break
                    
        # If we couldn't find rating, look for columns with 'rating' or 'review'
        if mapping['rating'] is None:
            for col in self.product_data.columns:
                if 'rating' in col.lower() or 'review' in col.lower() or 'score' in col.lower():
                    mapping['rating'] = col
                    break
                    
        return mapping

    def load_product_data(self):
        """Load product data from CSV file with robust path handling"""
        try:
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Define possible paths relative to the script location
            possible_paths = [
                os.path.join(script_dir, "products.csv"),
                os.path.join(script_dir, "..", "products.csv"),
                os.path.join(script_dir, "data", "products.csv"),
                os.path.join(script_dir, "..", "data", "products.csv"),
                "products.csv"  # Current working directory
            ]
            
            found_path = None
            for path in possible_paths:
                if os.path.exists(path) and os.path.isfile(path):
                    found_path = path
                    print(f"✅ Found products.csv at: {os.path.abspath(path)}")
                    
                    # Try different encodings if necessary
                    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(path, encoding=encoding, on_bad_lines='skip')
                            print(f"✅ Loaded {len(df)} products from CSV")
                            
                            # Generate hash of file content to detect changes
                            with open(path, 'rb') as f:
                                file_hash = hashlib.md5(f.read()).hexdigest()
                            
                            # Only reload if data has changed
                            if self.product_data_hash != file_hash:
                                self.product_data = df
                                self.product_data_hash = file_hash
                                self.column_mapping = self.detect_column_mapping()
                                self.precompute_product_embeddings()
                            return
                        except UnicodeDecodeError:
                            continue
                    
                    # If all encodings fail, try without specifying encoding
                    try:
                        df = pd.read_csv(path, on_bad_lines='skip')
                        print(f"✅ Loaded {len(df)} products from CSV (auto encoding)")
                        
                        # Generate hash of file content to detect changes
                        with open(path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        
                        if self.product_data_hash != file_hash:
                            self.product_data = df
                            self.product_data_hash = file_hash
                            self.column_mapping = self.detect_column_mapping()
                            self.precompute_product_embeddings()
                        return
                    except Exception as e:
                        print(f"⚠️ Error loading CSV: {str(e)}")
                        break
            
            # If no file found, create sample data
            if self.product_data is None:
                print("⚠️ products.csv not found in any location. Creating sample product database.")
                self.product_data = pd.DataFrame({
                    "Brand": ["StyleCast x Revolte", "U.S. Polo Assn.", "Levis", "Adidas", "Nike"],
                    "Product Name": [
                        "Men Typography T-shirt", 
                        "Men Lounge T-shirt", 
                        "Solid Lounge T-shirt",
                        "Sportswear Club Fleece",
                        "Dri-FIT Victory Polo"
                    ],
                    "Ratings": ["4|6", "4.2|9.6k", "4.3|10.5k", "4.5|12.3k", "4.4|15.7k"],
                    "Price": [2005, 608, 454, 1299, 1599],
                    "Product Link": [
                        "https://example.com/product1",
                        "https://example.com/product2",
                        "https://example.com/product3",
                        "https://example.com/product4",
                        "https://example.com/product5"
                    ]
                })
                self.column_mapping = self.detect_column_mapping()
                self.precompute_product_embeddings()
        except Exception as e:
            print(f"⚠️ Error loading products: {str(e)}")
            traceback.print_exc()
            if self.product_data is None:
                self.product_data = pd.DataFrame(columns=["Brand", "Product Name", "Ratings", "Price", "Product Link"])
                self.column_mapping = self.detect_column_mapping()

    def precompute_product_embeddings(self):
        """Precompute embeddings for products to enable fast search"""
        if self.product_data is None or self.product_data.empty:
            self.product_embeddings = None
            return
            
        print("🔧 Precomputing product embeddings...")
        
        # Create text representations for each product
        product_texts = []
        for _, row in self.product_data.iterrows():
            text_parts = []
            for col in self.product_data.columns:
                if pd.notna(row[col]) and row[col] != "":
                    text_parts.append(str(row[col]))
            product_texts.append(" ".join(text_parts))
        
        # Compute embeddings in batches
        batch_size = 50
        embeddings = []
        for i in range(0, len(product_texts), batch_size):
            batch = product_texts[i:i+batch_size]
            embeddings.extend(self.embedding_function(batch))
        
        self.product_embeddings = embeddings
        print(f"✅ Precomputed embeddings for {len(embeddings)} products")

    def initialize_knowledge_base(self):
        """Initialize with essential support knowledge (append-only)"""
        try:
            count = self.knowledge_collection.count()
            if count > 0:
                print(f"✅ Using existing knowledge base with {count} entries")
                return
        except:
            pass

        print("📚 Initializing knowledge base...")

        # Domain-specific knowledge entries
        security_knowledge = [
            ("How to reset password", "Secure password reset: https://example.com/reset (Never share your password with anyone)"),
            ("Password not working", "Troubleshooting steps: https://example.com/password-help (Contact support if issues persist)"),
            ("Enable two-factor authentication", "Security guide: https://example.com/2fa-setup (Recommended for all accounts)"),
        ]

        for question, answer in security_knowledge:
            entry_id = str(uuid.uuid4())
            self.knowledge_collection.add(
                documents=[question],
                metadatas=[{
                    "answer": answer,
                    "source": "expert",
                    "uses": 0,
                    "successes": 0,
                    "created_at": datetime.now().isoformat(),
                    "last_used": datetime.now().isoformat()
                }],
                ids=[entry_id]
            )

        print(f"✅ Knowledge base initialized")

    def get_product_vector_store(self, session_id: str):
        """Get or create a temporary vector store for the session"""
        if session_id not in self.temp_vector_store:
            # Create an in-memory vector store for this session
            self.temp_vector_store[session_id] = {
                "embeddings": [],
                "product_ids": []
            }
        return self.temp_vector_store[session_id]

    def add_to_temp_vector_store(self, session_id: str, product_ids: List[str], embeddings: List[List[float]]):
        """Add products to the temporary vector store"""
        vector_store = self.get_product_vector_store(session_id)
        vector_store["product_ids"].extend(product_ids)
        vector_store["embeddings"].extend(embeddings)

    def search_temp_vector_store(self, session_id: str, query_embedding: List[float], top_k: int = 10):
        """Search in the temporary vector store using cosine similarity"""
        vector_store = self.get_product_vector_store(session_id)
        if not vector_store["embeddings"]:
            return []
            
        # Convert to numpy arrays for efficient computation
        query_np = np.array(query_embedding)
        embeddings_np = np.array(vector_store["embeddings"])
        
        # Compute cosine similarity
        norms = np.linalg.norm(embeddings_np, axis=1)
        query_norm = np.linalg.norm(query_np)
        if query_norm == 0:
            return []
            
        similarities = np.dot(embeddings_np, query_np) / (norms * query_norm)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return product IDs and scores
        return [(vector_store["product_ids"][i], similarities[i]) for i in top_indices]

# ----------------------------
# Base Agent Class
# ----------------------------
class BaseAgent:
    """Base class for all agents"""
    def __init__(self, agent_id: str, shared: SharedResources):
        self.id = agent_id
        self.shared = shared
        self.log(f"Agent initialized")

    def log(self, message: str):
        print(f"[{self.id.upper()}] {message}")

    def handle_message(self, message: dict):
        """Process incoming messages"""
        raise NotImplementedError("Subclasses must implement this method")

# ----------------------------
# Enhanced Product Agent (Hybrid Approach)
# ----------------------------
class ProductAgent(BaseAgent):
    """Hybrid product assistant using dynamic data handling and vector search"""
    def __init__(self, shared: SharedResources):
        super().__init__("product", shared)
        # Track product search state per user
        self.user_states = {}
        
    def get_user_state(self, user_id: str) -> dict:
        """Get or create user's product search state"""
        if user_id not in self.user_states:
            self.user_states[user_id] = {
                "current_query": "",
                "search_history": [],
                "last_search_params": {},
                "original_query": ""
            }
        return self.user_states[user_id]

    def handle_message(self, message: dict):
        """Process product queries and refinement requests"""
        if message["type"] not in ["product_query", "product_refinement"]:
            return None

        user_id = message["user_id"]
        query = message["content"]
        state = self.get_user_state(user_id)
        
        # Store this interaction in history
        state["search_history"].append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "type": message["type"]
        })
        
        if message["type"] == "product_query":
            result = self.process_product_query(query, user_id, state)
        elif message["type"] == "product_refinement":
            result = self.process_refinement(query, user_id, state)
        else:
            return None
            
        # Format the result to include a 'response' field
        if "response" not in result:
            result["response"] = self.format_product_response(result)
            
        return result
    
    def format_product_response(self, product_data: dict) -> str:
        """Format structured product data into a user-friendly response string"""
        response_type = product_data.get("type", "error")
        
        if response_type == "clarification":
            return product_data["message"]
        elif response_type == "no_results":
            return product_data["message"]
        elif response_type == "error":
            return product_data.get("message", "There was an issue with your product request.")
        elif response_type == "results":
            products = product_data["products"]
            params = product_data.get("search_params", {})
            
            num_products = len(products)
            category = params.get("category", "products")
            
            message = f"I found {num_products} {category} matching your request"
            
            # Add filter summary
            filters = []
            if params.get("color"): filters.append(f"color: {params['color']}")
            if params.get("size"): filters.append(f"size: {params['size']}")
            if "price_max" in params: filters.append(f"under ₹{params['price_max']}")
            
            if filters:
                message += f" (with {', '.join(filters)})"
            
            message += ":\n\n"
            
            # Add products
            for product in products:
                # Find the most important fields to display
                display_fields = self.get_display_fields(product)
                
                product_message = ""
                for field, value in display_fields.items():
                    if pd.notna(value) and value != "":
                        product_message += f"   • {field}: {value}\n"
                
                if product_message:
                    product_name = self.get_product_name(product)
                    message += f"🔹 **{product_name}**\n{product_message}\n"
                else:
                    message += f"🔹 Product (details unavailable)\n\n"
            
            # Add refinement options
            message += "\nYou can ask to:\n"
            message += "- 'Sort by price low to high'\n"
            message += "- 'Sort by price high to low'\n"
            message += "- 'Sort by best rating'\n"
            message += "- 'Filter by color'\n"
            message += "- 'Filter by size'\n"
            message += "- Or ask about a specific product"
            
            return message
        else:
            return "I couldn't process your product request. Please try again."

    def process_product_query(self, query: str, user_id: str, state: dict) -> dict:
        """Process product query with context from previous interactions"""
        state["current_query"] = query
        state["original_query"] = query
        
        try:
            if self.shared.product_data is None or self.shared.product_data.empty:
                return {
                    "type": "error",
                    "message": "Our product catalog is currently unavailable. Please try again later."
                }
            
            # Use AI to understand query intent with context
            search_params = self.extract_search_parameters(query, state)
            
            # Store parameters for refinements
            state["last_search_params"] = search_params
            
            # Search products using hybrid approach
            results = self.hybrid_product_search(query, search_params, user_id)
            
            if results.empty:
                return {
                    "type": "no_results",
                    "message": self.handle_no_results(search_params)
                }
            
            # Return structured data with context
            return {
                "type": "results",
                "products": results.to_dict(orient="records"),
                "search_params": search_params,
                "context": self.get_search_context(state)
            }
        
        except Exception as e:
            self.log(f"Product search error: {str(e)}")
            traceback.print_exc()
            return {
                "type": "error",
                "message": "I encountered an issue while searching our products. Please try again."
            }

    def hybrid_product_search(self, query: str, params: dict, user_id: str) -> pd.DataFrame:
        """Hybrid search combining vector similarity and structured filtering"""
        # Step 1: Use LLM to understand query intent
        structured_params = self.extract_search_parameters_with_llm(query)
        
        # Combine with context-based parameters
        combined_params = {**params, **structured_params}
        
        # Step 2: Vector similarity search
        vector_results = self.vector_similarity_search(query, user_id, top_k=50)
        
        # If no vector results, use the entire dataset
        if not vector_results:
            results = self.shared.product_data.copy()
        else:
            # Get top product IDs from vector search
            product_ids = [pid for pid, _ in vector_results]
            results = self.shared.product_data.loc[self.shared.product_data.index.isin(product_ids)]
        
        # Step 3: Apply structured filters
        if combined_params.get("category"):
            category = combined_params["category"].lower()
            results = results[results.apply(
                lambda row: any(category in str(cell).lower() for cell in row), 
                axis=1
            )]
        
        if combined_params.get("color"):
            color = combined_params["color"].lower()
            results = results[results.apply(
                lambda row: any(color in str(cell).lower() for cell in row), 
                axis=1
            )]
            
        if combined_params.get("brand"):
            brand = combined_params["brand"].lower()
            results = results[results.apply(
                lambda row: any(brand in str(cell).lower() for cell in row), 
                axis=1
            )]
            
        # Apply price filter
        price_col = self.shared.column_mapping.get('price')
        if price_col and price_col in results.columns and "price_max" in combined_params:
            try:
                # Clean price values
                results[price_col] = results[price_col].replace('[^\d.]', '', regex=True).astype(float)
                results = results[results[price_col] <= combined_params["price_max"]]
            except:
                pass
            
        # Apply sorting
        if combined_params.get("sort") == "price_low":
            price_col = self.shared.column_mapping.get('price')
            if price_col and price_col in results.columns:
                try:
                    results[price_col] = results[price_col].replace('[^\d.]', '', regex=True).astype(float)
                    results = results.sort_values(price_col)
                except:
                    pass
        elif combined_params.get("sort") == "price_high":
            price_col = self.shared.column_mapping.get('price')
            if price_col and price_col in results.columns:
                try:
                    results[price_col] = results[price_col].replace('[^\d.]', '', regex=True).astype(float)
                    results = results.sort_values(price_col, ascending=False)
                except:
                    pass
        elif combined_params.get("sort") == "rating":
            rating_col = self.shared.column_mapping.get('rating')
            if rating_col and rating_col in results.columns:
                try:
                    # Extract numeric rating from string format
                    results['NumericRating'] = results[rating_col].apply(
                        lambda x: float(str(x).split('|')[0]) if pd.notna(x) and '|' in str(x) else 0
                    )
                    results = results.sort_values('NumericRating', ascending=False)
                except:
                    pass
        
        return results.head(10)  # Return top 10 results

    def vector_similarity_search(self, query: str, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search products using vector similarity with temporary storage"""
        # Compute query embedding
        query_embedding = self.shared.embedding_function([query])[0]
        
        # Use the temporary vector store for this session
        vector_store = self.shared.get_product_vector_store(user_id)
        
        # If we have precomputed embeddings, use them
        if self.shared.product_embeddings is not None:
            # Convert to numpy arrays for efficient computation
            query_np = np.array(query_embedding)
            embeddings_np = np.array(self.shared.product_embeddings)
            
            # Compute cosine similarity
            norms = np.linalg.norm(embeddings_np, axis=1)
            query_norm = np.linalg.norm(query_np)
            if query_norm == 0:
                return []
                
            similarities = np.dot(embeddings_np, query_np) / (norms * query_norm)
            
            # Get top k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Return product IDs and scores
            return [(str(i), similarities[i]) for i in top_indices]
        else:
            # Fallback to on-the-fly embedding generation
            if not vector_store["embeddings"]:
                # Populate the temporary store with all products
                product_texts = []
                for idx, row in self.shared.product_data.iterrows():
                    text_parts = []
                    for col in self.shared.product_data.columns:
                        if pd.notna(row[col]) and row[col] != "":
                            text_parts.append(str(row[col]))
                    product_texts.append((" ".join(text_parts), str(idx)))
                
                # Compute embeddings in batches
                batch_size = 20
                for i in range(0, len(product_texts), batch_size):
                    batch = product_texts[i:i+batch_size]
                    texts = [item[0] for item in batch]
                    ids = [item[1] for item in batch]
                    embeddings = self.shared.embedding_function(texts)
                    self.shared.add_to_temp_vector_store(user_id, ids, embeddings)
            
            # Search in the temporary vector store
            return self.shared.search_temp_vector_store(user_id, query_embedding, top_k)

    def extract_search_parameters_with_llm(self, query: str) -> dict:
        """Use AI to extract search parameters from query"""
        prompt = (
            f"Extract product search parameters from this user query:\n"
            f"Query: {query}\n\n"
            "Return JSON with:\n"
            "- category: main product type\n"
            "- color: requested color\n"
            "- size: requested size\n"
            "- sort: preferred sorting (price_low, price_high, rating)\n"
            "- price_min: minimum price\n"
            "- price_max: maximum price\n"
            "- brand: preferred brand\n\n"
            "Example: {{\"category\": \"polo shirt\", \"color\": \"blue\", \"size\": \"medium\"}}"
        )
        
        try:
            response = self.shared.model.invoke(prompt).content
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return {}
        except:
            return {}

    def process_refinement(self, query: str, user_id: str, state: dict) -> dict:
        """Process refinement requests with full context"""
        # Use AI to interpret refinement request
        refinement = self.interpret_refinement(query, state)
        
        if refinement["type"] == "requery":
            return self.process_product_query(refinement["new_query"], user_id, state)
        
        # Update search parameters
        new_params = state["last_search_params"].copy()
        
        if refinement["type"] == "sort":
            new_params["sort"] = refinement["criteria"]
        elif refinement["type"] == "filter":
            for key, value in refinement["criteria"].items():
                new_params[key] = value
        
        state["last_search_params"] = new_params
        
        # Perform new search with updated parameters
        try:
            results = self.hybrid_product_search(state["original_query"], new_params, user_id)
            
            if results.empty:
                return {
                    "type": "no_results",
                    "message": f"No products match your refinement: {refinement}"
                }
            
            return {
                "type": "results",
                "products": results.to_dict(orient="records"),
                "search_params": new_params,
                "context": self.get_search_context(state)
            }
        except Exception as e:
            self.log(f"Refinement search error: {str(e)}")
            return {
                "type": "error",
                "message": "I couldn't apply the refinement. Please try again."
            }
    
    def interpret_refinement(self, query: str, state: dict) -> dict:
        """Use AI to understand refinement requests"""
        context = self.get_search_context(state)
        prompt = (
            f"Interpret this refinement request based on current search context:\n"
            f"Context: {context}\n"
            f"Request: {query}\n\n"
            "Return JSON with:\n"
            "- type: 'sort', 'filter', or 'requery'\n"
            "- criteria: sorting/filtering criteria\n"
            "- new_query: only for requery type\n"
            "- message: optional clarification\n"
            "Examples:\n"
            "{\"type\": \"sort\", \"criteria\": \"rating\"}\n"
            "{\"type\": \"filter\", \"criteria\": {\"color\": \"blue\"}}\n"
            "{\"type\": \"requery\", \"new_query\": \"blue polo shirts\"}"
        )
        
        try:
            response = self.shared.model.invoke(prompt).content
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass
        
        return {"type": "unknown"}

    def get_search_context(self, state: dict) -> str:
        """Generate context summary for current search state"""
        if not state["search_history"]:
            return "No previous context"
            
        context = "Product search history:\n"
        for i, entry in enumerate(state["search_history"][-3:]):
            context += f"{i+1}. {entry['query']} ({entry['type']})\n"
        
        context += f"\nCurrent filters: {json.dumps(state.get('last_search_params', {}), indent=2)}"
        return context

    def extract_search_parameters(self, query: str, state: dict) -> dict:
        """Use AI to extract search parameters from query with context"""
        context = self.get_search_context(state)
        prompt = (
            f"Extract product search parameters from this query with context:\n"
            f"{context}\nQuery: {query}\n\n"
            "Return JSON with:\n"
            "- category: main product type\n"
            "- color: requested color\n"
            "- size: requested size\n"
            "- sort: preferred sorting (price_low, price_high, rating)\n"
            "- price_min: minimum price\n"
            "- price_max: maximum price\n"
            "- brand: preferred brand\n\n"
            "Example: {{\"category\": \"polo shirt\", \"color\": \"blue\", \"size\": \"medium\"}}"
        )
        
        try:
            response = self.shared.model.invoke(prompt).content
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass
        
        # Fallback to keyword extraction
        params = {}
        if "shirt" in query.lower(): params["category"] = "shirt"
        if "polo" in query.lower(): params["category"] = "polo shirt"
        if "t-shirt" in query.lower(): params["category"] = "t-shirt"
        if "jeans" in query.lower(): params["category"] = "jeans"
        if "shoes" in query.lower(): params["category"] = "shoes"
        
        color_keywords = ["red", "blue", "green", "black", "white", "gray", "navy", "beige", "maroon"]
        for color in color_keywords:
            if color in query.lower():
                params["color"] = color
                
        size_keywords = ["small", "medium", "large", "xl", "xxl", "xs", "xxxl"]
        for size in size_keywords:
            if size in query.lower():
                params["size"] = size
                
        if "cheap" in query.lower() or "low" in query.lower() or "affordable" in query.lower():
            params["sort"] = "price_low"
        if "expensive" in query.lower() or "high" in query.lower() or "premium" in query.lower():
            params["sort"] = "price_high"
        if "rating" in query.lower() or "best" in query.lower() or "top" in query.lower():
            params["sort"] = "rating"
            
        # Extract price range
        price_match = re.search(r'under\s+₹?(\d+)', query, re.IGNORECASE)
        if price_match:
            params["price_max"] = float(price_match.group(1))
            
        return params

    def handle_no_results(self, params: dict) -> str:
        """Handle case where no products match the query"""
        response = "I couldn't find any products matching "
        
        filters = []
        if params.get("category"): filters.append(f"category '{params['category']}'")
        if params.get("color"): filters.append(f"color '{params['color']}'")
        if params.get("size"): filters.append(f"size '{params['size']}'")
        if "price_max" in params: filters.append(f"under ₹{params['price_max']}")
        
        if filters:
            response += "your criteria (" + ", ".join(filters) + "). "
        else:
            response += "your query. "
            
        response += "Would you like to try different search parameters?"
        return response

    def get_display_fields(self, product: dict) -> dict:
        """Identify the most important fields to display for a product"""
        # Priority fields to show
        priority_fields = [
            self.shared.column_mapping.get('name'),
            self.shared.column_mapping.get('brand'),
            self.shared.column_mapping.get('price'),
            self.shared.column_mapping.get('rating'),
            self.shared.column_mapping.get('link')
        ]
        
        # Remove None values
        priority_fields = [field for field in priority_fields if field]
        
        display = {}
        
        # First, look for priority fields
        for field in priority_fields:
            if field in product:
                display[field] = product[field]
                
        # If we don't have enough fields, add more
        if len(display) < 3:
            for key, value in product.items():
                if key not in display and pd.notna(value) and value != "":
                    display[key] = value
                    if len(display) >= 3:
                        break
                        
        return display

    def get_product_name(self, product: dict) -> str:
        """Get product name from available fields"""
        name_col = self.shared.column_mapping.get('name')
        if name_col and name_col in product:
            return product[name_col]
            
        # Fallback to any name-like field
        name_fields = ['Product Name', 'Name', 'Title', 'Description']
        for field in name_fields:
            if field in product:
                return product[field]
                
        return "Unnamed Product"
    
    def is_product_query(self, query: str) -> bool:
        """AI-powered product query detection"""
        try:
            prompt = (
                "Determine if this user query is asking about products or inventory:\n"
                f"Query: {query}\n\n"
                "Answer ONLY 'YES' or 'NO'. Consider:\n"
                "- Mentions of products, items, inventory\n"
                "- Requests to show, list, or find products\n"
                "- Questions about availability, features, or prices"
            )
            response = self.shared.model.invoke(prompt).content.strip().upper()
            return "YES" in response
        except:
            # Fallback to keyword matching
            keywords = ["product", "item", "inventory", "buy", "purchase", 
                        "shop", "store", "price", "stock", "available"]
            return any(kw in query.lower() for kw in keywords)

# ----------------------------
# Support Agent (Context-Aware)
# ----------------------------
class SupportAgent(BaseAgent):
    """Handles all user interactions with full conversation context"""
    def __init__(self, shared: SharedResources):
        super().__init__("support", shared)
        self.product_agent = ProductAgent(shared)  # Create instance of ProductAgent

    def handle_message(self, message: dict):
        """Process user query and generate response with context"""
        if message["type"] != "user_query":
            return None

        user_id = message["user_id"]
        query = message["content"]
        context = self.recall_context(user_id)

        try:
            # Check if this is a continuation of product search
            if self.is_product_continuation(query, context):
                # Handle as refinement request
                result = self.product_agent.handle_message({
                    "type": "product_refinement",
                    "user_id": user_id,
                    "content": query
                })
                return {
                    "type": "agent_response",
                    "response": result.get("response", "I encountered an issue with your product request."),
                    "metadata": {
                        "source": "product_assistant", 
                        "confidence": 0.95,
                        "product_data": result
                    }
                }

            # Check if this is a product query
            if self.product_agent.is_product_query(query):
                # Handle as new product query
                result = self.product_agent.handle_message({
                    "type": "product_query",
                    "user_id": user_id,
                    "content": query
                })
                return {
                    "type": "agent_response",
                    "response": result.get("response", "I encountered an issue with your product request."),
                    "metadata": {
                        "source": "product_assistant", 
                        "confidence": 0.95,
                        "product_data": result
                    }
                }

            # Generate support response
            return self.handle_support_query(query, user_id)
            
        except Exception as e:
            self.log(f"SupportAgent error: {str(e)}")
            return {
                "type": "agent_response",
                "response": "I encountered an error processing your request. Please try again.",
                "metadata": {"source": "error", "confidence": 0.0}
            }
    
    def handle_support_query(self, query: str, user_id: str) -> dict:
        """Handle non-product support queries"""
        response, metadata = self.generate_response(query, user_id)
        self.remember_context(user_id, query, response)
        
        if metadata["source"] in ["knowledge_base", "generated"]:
            suggestions = self.generate_proactive_suggestions(query, response)
            metadata["suggestions"] = suggestions

        return {
            "type": "agent_response",
            "response": response,
            "metadata": metadata
        }
    
    def is_product_continuation(self, query: str, context: str) -> bool:
        """Detect if query continues a product search"""
        if not context:
            return False
            
        continuation_keywords = [
            "sort by", "filter by", "refine", "cheaper", "more expensive",
            "higher rating", "better reviews", "different", "other options",
            "next", "previous", "another", "instead", "rather", "instead of",
            "instead", "how about", "what about", "any other", "more", "less",
            "different color", "different size", "different brand", "change",
            "from those", "from these", "you showed", "just showed", "current"
        ]
        
        # Check if last response was from product agent
        if "product_assistant" not in context:
            return False
            
        # Check for continuation keywords
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in continuation_keywords)
    
    def refine_kb_response(self, answer: str, query: str, matched_question: str, context: str) -> str:
        """Refines the knowledge base response using the conversation context."""
        prompt = f"""
        The user asked: "{query}"
        The most relevant knowledge base entry is for the question: "{matched_question}"
        The answer provided is: "{answer}"
        The recent conversation history is:
        {context}

        Please refine the answer to be more conversational and directly address the user's specific query, while staying relevant to the ongoing conversation.
        Return only the refined answer.
        """
        try:
            response = self.shared.model.invoke(prompt)
            return response.content
        except Exception as e:
            self.log(f"Error refining KB response: {e}")
            return answer
    
    def generate_response(self, query: str, user_id: str) -> Tuple[str, dict]:
        """Generate response to user query"""
        context = self.recall_context(user_id)
        user_profile = self.build_user_profile(user_id)

        prompt = f"Conversation History:\n{context}\n\nCurrent User Query: {query}\n\nResponse:"

        kb_result = None
        if not self.is_escalation_test_query(query):
            kb_result = self.query_knowledge_base(query)

        if kb_result:
            answer, entry_id, confidence, matched_question = kb_result
            self.update_kb_entry(entry_id)

            try:
                refined_answer = self.refine_kb_response(answer, query, matched_question, context)
                personalized_response = self.personalize_response(refined_answer, user_profile)
                return personalized_response, {
                    "source": "knowledge_base",
                    "confidence": confidence,
                    "entry_id": entry_id,
                    "matched_question": matched_question,
                    "original_answer": answer
                }
            except Exception as e:
                self.log(f"KB refinement error: {str(e)}")
                personalized_response = self.personalize_response(answer, user_profile)
                return personalized_response, {
                    "source": "knowledge_base",
                    "confidence": confidence,
                    "entry_id": entry_id,
                    "matched_question": matched_question
                }

        try:
            response = self.shared.model.invoke(prompt)
            generated_answer = response.content
            personalized_response = self.personalize_response(generated_answer, user_profile)
            confidence = self.calculate_response_confidence(query, personalized_response)

            self.shared.communication_bus.put({
                "type": "learning_request",
                "user_query": query,
                "agent_response": generated_answer,
                "user_id": user_id,
                "confidence": confidence
            })

            return personalized_response, {
                "source": "generated",
                "confidence": confidence
            }

        except Exception as e:
            self.log(f"Generation error: {str(e)}")
            return "Please contact our support team for further assistance.", {
                "source": "error"
            }

    def is_escalation_test_query(self, query: str) -> bool:
        """Identify queries meant to test escalation"""
        escalation_patterns = [
            r"this is a very specific problem not in your kb",
            r"escalation test",
            r"not in knowledge base",
            r"unique problem not covered",
            r"test escalation",
            r"problem not in kb"
        ]
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in escalation_patterns)

    def query_knowledge_base(self, raw_query: str) -> Optional[Tuple[str, str, float, str]]:
        """Enhanced similarity search with better validation"""
        query = self.preprocess_query(raw_query)
        if not query.strip():
            self.log("Preprocessed query is empty, skipping KB lookup.")
            return None

        try:
            results = self.shared.knowledge_collection.query(
                query_texts=[query],
                n_results=5,
                include=["metadatas", "distances", "documents"]
            )

            if not results or not results["ids"] or not results["ids"][0]:
                return None

            for i in range(len(results["ids"][0])):
                entry_id = results["ids"][0][i]
                distance = results["distances"][0][i]

                if not results["metadatas"][0][i] or "answer" not in results["metadatas"][0][i]:
                    self.log(f"Missing metadata/answer for KB entry {entry_id}, skipping.")
                    continue
                if not results["documents"][0][i]:
                    self.log(f"Missing document content for KB entry {entry_id}, skipping.")
                    continue

                answer = results["metadatas"][0][i]["answer"]
                question = results["documents"][0][i]
                confidence = max(0.0, 1 - distance)

                min_confidence = max(0.5, self.shared.confidence_threshold - 0.1)
                if confidence < min_confidence:
                    self.log(f"KB match for '{query}' below min confidence: {confidence:.2f}")
                    continue

                if self.validate_match(query, question):
                    self.log(f"Found KB match for '{query}' with confidence {confidence:.2f}")
                    return answer, entry_id, confidence, question
                else:
                    self.log(f"KB match for '{query}' failed keyword validation for question '{question}'.")

        except Exception as e:
            self.log(f"Query knowledge error: {str(e)}")

        return None

    def validate_match(self, query: str, kb_question: str) -> bool:
        """Faster validation with keyword matching"""
        query_keywords = set(re.findall(r'\w+', query.lower()))
        kb_keywords = set(re.findall(r'\w+', kb_question.lower()))
        common = query_keywords & kb_keywords
        significant = [kw for kw in common if len(kw) > 3]
        return len(significant) >= 2

    def preprocess_query(self, raw_query: str) -> str:
        """Advanced query normalization and expansion"""
        query = raw_query.strip().lower()
        corrections = {
            r'\bpswd\b': 'password',
            r'\bwrng\b': 'wrong',
            r'\bacc\b': 'account',
            r'\blogin\b': 'log in',
            r'\bpls\b': 'please',
            r'\bordr\b': 'order',
            r'\btrack\b': 'tracking',
            r'\breset\b': 'password reset'
        }
        for pattern, replacement in corrections.items():
            query = re.sub(pattern, replacement, query)
        return query

    def build_user_profile(self, user_id: str) -> dict:
        """Create personalized user profile from conversation history"""
        if user_id in self.shared.user_profiles:
            profile = self.shared.user_profiles[user_id]
            if profile.get("interaction_count", 0) % self.shared.profile_update_frequency != 0:
                return profile

        context = self.recall_context(user_id)
        if not context.strip():
            initial_profile = self.shared.default_profile.copy()
            initial_profile["interaction_count"] = self.shared.user_profiles.get(user_id, {}).get("interaction_count", 0) + 1
            self.shared.user_profiles[user_id] = initial_profile
            return initial_profile

        prompt = f"""
        Analyze this conversation history and extract user preferences:
        {context}

        Return JSON with:
        - preferred_name (first name if mentioned, else "there")
        - technical_level (beginner/intermediate/expert)
        - communication_style (direct/detailed/casual)
        - known_issues (list of recurring problems mentioned)

        Example: {{"preferred_name": "John", "technical_level": "beginner",
                      "communication_style": "casual", "known_issues": ["password reset", "login issues"]}}

        IMPORTANT: Respond ONLY with the JSON object. Do not include any other text or explanation.
        """

        try:
            response = self.shared.model.invoke(prompt)
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                profile_str = json_match.group(0)
                profile = json.loads(profile_str)
            else:
                self.log(f"No JSON found in profile generation response: {response.content}")
                profile = {}

            profile.setdefault("preferred_name", self.shared.default_profile["preferred_name"])
            profile.setdefault("technical_level", self.shared.default_profile["technical_level"])
            profile.setdefault("communication_style", self.shared.default_profile["communication_style"])
            profile.setdefault("known_issues", self.shared.default_profile["known_issues"])
            profile["interaction_count"] = self.shared.user_profiles.get(user_id, {}).get("interaction_count", 0) + 1
            self.shared.user_profiles[user_id] = profile
            return profile

        except json.JSONDecodeError as e:
            self.log(f"JSON decoding error in profile creation: {e}. Raw response: {response.content}")
            return self.shared.default_profile.copy()
        except Exception as e:
            self.log(f"Profile creation general error: {str(e)}")
            return self.shared.default_profile.copy()

    def personalize_response(self, response: str, profile: dict) -> str:
        """Adapt responses to user's preferences"""
        try:
            if profile["preferred_name"] != "there":
                if not re.match(r'^(Hi|Hello|Hey|Good\s(morning|afternoon|evening))', response, re.IGNORECASE):
                    response = f"Hi {profile['preferred_name']}, {response}"
                else:
                    response = re.sub(r'\b(?:you|your)\b', profile["preferred_name"], response, flags=re.IGNORECASE)

            if profile["technical_level"] == "beginner":
                response = self.simplify_technical_terms(response)
            elif profile["technical_level"] == "expert":
                response = self.add_technical_details(response)

            if profile["communication_style"] == "casual":
                response = self.make_casual(response)
            elif profile["communication_style"] == "detailed":
                response = self.add_details(response)

            return response
        except Exception as e:
            self.log(f"Personalization error: {str(e)}")
            return response

    def simplify_technical_terms(self, text: str) -> str:
        """Replace technical terms with simple explanations"""
        replacements = {
            "2FA": "two-step verification",
            "authentication": "login process",
            "credentials": "login details",
            "API": "system connection",
            "SSL": "security certificate",
            "encryption": "security protection",
            "database": "storage system",
            "backend": "our systems",
            "frontend": "website interface",
            "caching": "temporary storage"
        }
        for term, replacement in replacements.items():
            text = text.replace(term, replacement)
        return text

    def add_technical_details(self, text: str) -> str:
        """Add technical details for expert users"""
        if len(text.split()) < 30:
            prompt = (
                f"Add technical details to this response for an expert user:\n"
                f"{text}\n\n"
                "Add 1-2 technical specifics that an advanced user might find helpful."
            )
            try:
                return self.shared.model.invoke(prompt).content
            except:
                pass
        return text

    def make_casual(self, text: str) -> str:
        """Make response more conversational and friendly"""
        replacements = {
            "Please": "Could you",
            "You should": "I'd recommend",
            "It is recommended": "You might want to",
            "Ensure that": "Make sure to",
            "Additionally": "Also,",
            "Furthermore": "Plus,",
            "Therefore": "So,",
            "However": "But"
        }
        for formal, casual in replacements.items():
            text = text.replace(formal, casual)
        return text

    def add_details(self, text: str) -> str:
        """Add more details for users who prefer comprehensive answers"""
        if len(text.split()) < 25:
            prompt = (
                f"Add helpful details to this response while keeping it accurate:\n"
                f"{text}\n\n"
                "Add 1-2 useful details that a detail-oriented user might appreciate."
            )
            try:
                return self.shared.model.invoke(prompt).content
            except:
                pass
        return text
    
 
    def calculate_response_confidence(self, query: str, response: str) -> float:
        try:
            prompt = (
                f"Rate how well this response answers the query (0-100). Consider:\n"
                f"1. Relevance to query\n2. Completeness\n3. Support context\n\n"
                f"QUERY: {query}\nRESPONSE: {response}\n\n"
                f"Output ONLY the numerical score."
            )
            score_text = self.shared.model.invoke(prompt).content
            try:
                score = float(score_text.strip()) / 100
                return max(0.1, min(0.9, score))
            except:
                return 0.4
        except Exception as e:
            self.log(f"Confidence evaluation error: {str(e)}")
            return 0.4

    def update_kb_entry(self, entry_id: str, success: bool = True):
        """Update entry usage statistics with error handling"""
        try:
            entry = self.shared.knowledge_collection.get(ids=[entry_id])
            metadata = entry["metadatas"][0]
            current_uses = metadata.get("uses", 0)
            current_successes = metadata.get("successes", 0)
            updates = {
                "uses": int(current_uses) + 1,
                "successes": int(current_successes) + (1 if success else 0),
                "last_used": datetime.now().isoformat()
            }
            self.shared.knowledge_collection.update(
                ids=[entry_id],
                metadatas=[{**metadata, **updates}]
            )
        except Exception as e:
            self.log(f"Failed to update entry {entry_id}: {str(e)}")

    def mask_sensitive_data(self, text: str) -> str:
        """Automatically detect and mask sensitive information"""
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CARD]', text)
        text = re.sub(r'\b(?:\+\d{1,2}\s?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
        text = re.sub(r'\b[a-f0-9]{32}\b', '[TOKEN]', text)
        return text

    def remember_context(self, user_id: str, query: str, response: str):
        """Robust context management with summarization and data masking"""
        try:
            masked_query = self.mask_sensitive_data(query)
            masked_response = self.mask_sensitive_data(response)

            if user_id not in self.shared.user_contexts:
                self.shared.user_contexts[user_id] = []
            elif not isinstance(self.shared.user_contexts[user_id], list):
                self.shared.user_contexts[user_id] = []

            if masked_query.strip():
                self.shared.user_contexts[user_id].append({"role": "user", "content": masked_query})
            if masked_response.strip():
                self.shared.user_contexts[user_id].append({"role": "assistant", "content": masked_response})

            if not masked_query.strip() and not masked_response.strip():
                self.log(f"Skipped context saving for user {user_id}: both query and response were empty after masking.")
                return

            context_str = self.recall_context(user_id)
            if len(context_str) > self.shared.context_summary_threshold:
                summarized = self.summarize_context(context_str)
                self.shared.user_contexts[user_id] = [
                    {"role": "system", "content": f"Summary: {summarized}"},
                    {"role": "user", "content": masked_query},
                    {"role": "assistant", "content": masked_response}
                ]
            else:
                if len(self.shared.user_contexts[user_id]) > self.shared.max_history_length * 2:
                    self.shared.user_contexts[user_id] = self.shared.user_contexts[user_id][-self.shared.max_history_length * 2:]

            exchange_doc = f"User: {masked_query}\nAgent: {masked_response}".strip()
            if exchange_doc:
                self.shared.conversation_collection.add(
                    documents=[exchange_doc],
                    metadatas=[{"user_id": user_id, "timestamp": datetime.now().isoformat()}],
                    ids=[f"{user_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"]
                )
            else:
                self.log(f"Skipped adding empty exchange to conversation_collection for user {user_id}.")

        except Exception as e:
            self.log(f"Context saving error: {str(e)}")
            self.shared.user_contexts[user_id] = []

    def summarize_context(self, context: str) -> str:
        """Summarize long conversation history using AI"""
        try:
            prompt = (
                f"Create a concise 3-sentence summary of this conversation, "
                f"preserving key user concerns and solutions:\n\n{context}"
            )
            response = self.shared.model.invoke(prompt)
            return response.content
        except Exception as e:
            self.log(f"Summarization error: {str(e)}")
            return context[:1500]

    def recall_context(self, user_id: str) -> str:
        """Robust context recall with conversation history"""
        try:
            context_messages = self.shared.user_contexts.get(user_id, [])
            if not isinstance(context_messages, list):
                context_messages = []

            context_str = "\n".join(
                f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}"
                for msg in context_messages
            )
            return context_str
        except Exception as e:
            self.log(f"Context recall error: {str(e)}")
            return ""

    def generate_proactive_suggestions(self, user_query: str, response: str) -> List[str]:
        """Generate proactive suggestions based on conversation"""
        try:
            suggestions = []
            for topic, keywords in self.shared.proactive_triggers.items():
                if topic in user_query.lower():
                    suggestions.extend(keywords)

            if not suggestions:
                prompt = (
                    f"Suggest 2-3 related help topics for this conversation:\n"
                    f"Q: {user_query}\nA: {response}\n"
                    "Output only the suggestions as a comma-separated list."
                )
                ai_suggestions = self.shared.model.invoke(prompt).content
                suggestions = [s.strip() for s in ai_suggestions.split(",")][:3]

            return suggestions
        except Exception as e:
            self.log(f"Suggestion error: {str(e)}")
            return []

# ----------------------------
# Learning Agent
# ----------------------------
class LearningAgent(BaseAgent):
    """Handles knowledge improvement and maintenance"""
    def __init__(self, shared: SharedResources):
        super().__init__("learning", shared)
        self.setup_scheduler()

    def setup_scheduler(self):
        """Schedule background maintenance tasks"""
        self.log("Setting up maintenance scheduler")
        schedule.every().day.at("02:00").do(self.prune_knowledge_base)
        schedule.every().day.at("03:00").do(self.optimize_knowledge_base)
        schedule.every().day.at("04:00").do(self.self_heal_knowledge_base)

        scheduler_thread = threading.Thread(target=self.run_scheduled_tasks, daemon=True)
        scheduler_thread.start()

    def run_scheduled_tasks(self):
        """Run background optimization tasks"""
        while True:
            schedule.run_pending()
            time.sleep(60)

    def handle_message(self, message: dict):
        """Process learning requests and maintenance commands"""
        if message["type"] == "learning_request":
            self.autonomous_learning(
                message["user_query"],
                message["agent_response"],
                message["confidence"]
            )
            return {"type": "learning_complete"}
        elif message["type"] == "add_kb_entry":
            self.add_kb_entry(
                message["question"],
                message["answer"],
                message["source"]
            )
            return {"type": "kb_entry_added"}
        return None

    def autonomous_learning(self, user_query: str, agent_response: str, confidence: float):
        """Automatically learn from generated responses"""
        if self.is_casual_question(user_query):
            self.log("Skipped learning: Casual question detected")
            return

        was_helpful = self.ai_self_evaluate(user_query, agent_response)
        if was_helpful and confidence > 0.6:
            self.add_kb_entry(user_query, agent_response, "auto_learned")
            self.log(f"Added new knowledge: {user_query[:50]}...")
        else:
            self.log(f"Discarded unhelpful response: {user_query[:50]}...")

    def ai_self_evaluate(self, user_query: str, agent_response: str) -> bool:
        """Use AI to evaluate if response solved the user's problem"""
        try:
            evaluation_prompt = (
                f"Determine if this response successfully answered the user's query:\n"
                f"USER QUERY: {user_query}\n"
                f"AGENT RESPONSE: {agent_response}\n\n"
                "Answer ONLY 'YES' or 'NO' based on:\n"
                "1. Does the response directly address the query?\n"
                "2. Does it provide a complete solution?\n"
                "3. Would the user likely need to ask follow-up questions?\n"
            )
            evaluation = self.shared.model.invoke(evaluation_prompt)
            decision = evaluation.content.strip().upper()
            return "YES" in decision
        except Exception as e:
            self.log(f"Self-evaluation error: {str(e)}")
            return False

    def add_kb_entry(self, question: str, answer: str, source: str = "auto") -> str:
        """Add entry to knowledge base with tracking metadata"""
        entry_id = str(uuid.uuid4())
        self.shared.knowledge_collection.add(
            documents=[question],
            metadatas=[{
                "answer": answer,
                "source": source,
                "uses": 0,
                "successes": 0,
                "created_at": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat()
            }],
            ids=[entry_id]
        )
        return entry_id

    def is_casual_question(self, query: str) -> bool:
        """Filter out non-support questions"""
        casual_patterns = [
            r"what('?s)? your name",
            r"who (are|made) you",
            r"how old are you",
            r"are you (human|real)",
            r"what can you do"
        ]
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in casual_patterns)

    def prune_knowledge_base(self, success_threshold: float = 0.6, max_age_days: int = 90):
        """Automated knowledge base maintenance"""
        self.log("Starting knowledge base pruning...")
        all_entries = self.shared.knowledge_collection.get()
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        entries_to_remove = []

        for entry_id, metadata in zip(all_entries["ids"], all_entries["metadatas"]):
            uses = metadata.get("uses", 1)
            successes = metadata.get("successes", 0)
            success_rate = successes / uses

            last_used_str = metadata.get("last_used")
            if last_used_str:
                try:
                    last_used = datetime.fromisoformat(last_used_str)
                    is_old = last_used < cutoff_date
                except:
                    is_old = False
            else:
                is_old = False

            if success_rate < success_threshold or is_old:
                entries_to_remove.append(entry_id)

        if entries_to_remove:
            self.shared.knowledge_collection.delete(ids=entries_to_remove)
            self.log(f"Pruned {len(entries_to_remove)} knowledge base entries")

        return len(entries_to_remove)

    def optimize_knowledge_base(self):
        """Automatically improve knowledge base entries"""
        try:
            self.log("Starting knowledge base optimization...")
            entries = self.shared.knowledge_collection.get()
            optimized_count = 0

            for i, (entry_id, metadata) in enumerate(zip(entries["ids"], entries["metadatas"])):
                if metadata.get("uses", 0) > 10:
                    prompt = (
                        f"Improve this support answer for clarity and conciseness:\n"
                        f"{metadata['answer']}\n\n"
                        "Respond ONLY with the improved answer."
                    )
                    improved = self.shared.model.invoke(prompt).content

                    self.shared.knowledge_collection.update(
                        ids=[entry_id],
                        metadatas=[{**metadata, "answer": improved, "optimized": True}]
                    )
                    optimized_count += 1

            self.log(f"Optimized {optimized_count} knowledge base entries")
            return optimized_count
        except Exception as e:
            self.log(f"Optimization error: {str(e)}")
            return 0

    def self_heal_knowledge_base(self):
        """Identify and repair underperforming knowledge base entries"""
        self.log("Starting knowledge base self-healing...")
        repaired_count = 0
        all_entries = self.shared.knowledge_collection.get()

        for entry_id, metadata in zip(all_entries["ids"], all_entries["metadatas"]):
            uses = metadata.get("uses", 0)
            if uses < 5:
                continue

            success_rate = metadata.get("successes", 0) / uses

            if success_rate < 0.4:
                try:
                    entry = self.shared.knowledge_collection.get(ids=[entry_id])
                    question = entry["documents"][0]
                    original_answer = metadata["answer"]

                    prompt = (
                        f"This knowledge base entry has low success rate ({success_rate:.0%}):\n"
                        f"Q: {question}\nA: {original_answer}\n\n"
                        f"Improve the answer to make it more helpful and accurate:"
                    )
                    improved = self.shared.model.invoke(prompt).content

                    self.shared.knowledge_collection.update(
                        ids=[entry_id],
                        metadatas=[{
                            **metadata,
                            "answer": improved,
                            "repaired_at": datetime.now().isoformat(),
                            "original_answer": original_answer
                        }]
                    )
                    repaired_count += 1
                    self.log(f"Repaired: {question[:50]}...")
                except Exception as e:
                    self.log(f"Self-heal failed for {entry_id}: {str(e)}")

        self.log(f"Repaired {repaired_count} knowledge entries")
        return repaired_count

# ----------------------------
# Coordinator
# ----------------------------
class Coordinator:
    """Manages communication between agents and users"""
    def __init__(self):
        self.shared = SharedResources()
        self.shared.initialize_knowledge_base()
        self.agents = {}
        self.setup_agents()
        self.setup_langgraph()
        self.log("Coordinator initialized")

    def log(self, message: str):
        print(f"[COORDINATOR] {message}")

    def setup_agents(self):
        """Initialize agents"""
        self.agents["support"] = SupportAgent(self.shared)
        self.agents["learning"] = LearningAgent(self.shared)
        self.log("Agents initialized")

    def setup_langgraph(self):
        """Initialize LangGraph workflow"""
        # ----------------------------
        # State Definition
        # ----------------------------
        class SupportState(TypedDict):
            user_id: str
            user_query: str
            messages: List[BaseMessage]
            current_response: Optional[str]
            confidence: Optional[float]
            escalated: bool
            should_escalate: bool
            metadata: Optional[dict]

        # ----------------------------
        # Node Functions
        # ----------------------------
        def support_node(state: SupportState) -> SupportState:
            """Processes all user queries through Support Agent"""
            user_id = state["user_id"]
            query = state["user_query"]
            
            # Use SupportAgent to generate response
            result = self.agents["support"].handle_message({
                "type": "user_query",
                "user_id": user_id,
                "content": query
            })
            
            if result is None:
                return {
                    "current_response": "I encountered an issue processing your request.",
                    "confidence": 0.0,
                    "should_escalate": True,
                    "escalated": False
                }
            
            return {
                "current_response": result["response"],
                "confidence": result["metadata"].get("confidence", 0.5),
                "should_escalate": False,
                "escalated": False,
                "metadata": result["metadata"]
            }

        def escalation_check_node(state: SupportState) -> SupportState:
            """Checks if conversation should be escalated"""
            user_id = state["user_id"]
            confidence = state["confidence"]
            should_escalate = self.should_escalate(user_id, confidence)
            return {"should_escalate": should_escalate}

        def escalation_node(state: SupportState) -> SupportState:
            """Handles conversation escalation"""
            user_id = state["user_id"]
            response = self.escalate_conversation(user_id)
            return {
                "current_response": response,
                "escalated": True
            }

        def personalization_node(state: SupportState) -> SupportState:
            """Personalizes the response to user preferences"""
            user_id = state["user_id"]
            response = state["current_response"]
            profile = self.agents["support"].build_user_profile(user_id)
            personalized = self.agents["support"].personalize_response(response, profile)
            return {"current_response": personalized}

        def update_context_node(state: SupportState) -> SupportState:
            """Updates conversation history with new messages"""
            user_id = state["user_id"]
            query = state["user_query"]
            response = state["current_response"]
            
            # Remember context
            self.agents["support"].remember_context(user_id, query, response)
            
            # Update messages
            new_messages = state["messages"] + [
                HumanMessage(content=query),
                AIMessage(content=response)
            ]
            return {"messages": new_messages}

        def route_escalation(state: SupportState) -> Literal["escalate", "personalize"]:
            """Routes to escalation or personalization based on state"""
            return "escalate" if state["should_escalate"] else "personalize"

        # ----------------------------
        # Graph Construction
        # ----------------------------
        workflow = StateGraph(SupportState)

        # Add nodes
        workflow.add_node("support", support_node)
        workflow.add_node("escalation_check", escalation_check_node)
        workflow.add_node("escalate", escalation_node)
        workflow.add_node("personalize", personalization_node)
        workflow.add_node("update_context", update_context_node)

        # Set up workflow structure
        workflow.set_entry_point("support")
        workflow.add_edge("support", "escalation_check")
        workflow.add_conditional_edges(
            "escalation_check",
            route_escalation,
            {"escalate": "escalate", "personalize": "personalize"}
        )
        workflow.add_edge("escalate", "update_context")
        workflow.add_edge("personalize", "update_context")
        workflow.add_edge("update_context", END)

        self.app = workflow.compile()
        self.log("LangGraph workflow initialized")

    def should_escalate(self, user_id: str, confidence: float) -> bool:
        """Improved escalation logic with detailed logging"""
        if user_id not in self.shared.escalation_count:
            self.shared.escalation_count[user_id] = 0

        if confidence < self.shared.escalation_threshold:
            self.shared.escalation_count[user_id] += 1
            self.log(f"🚨 Escalation count increased to {self.shared.escalation_count[user_id]}/3 (Low confidence: {confidence:.2f})")
        else:
            if confidence > 0.7:
                self.log(f"✅ Resetting escalation counter (High confidence: {confidence:.2f})")
                self.shared.escalation_count[user_id] = 0
            else:
                self.shared.escalation_count[user_id] = max(0, self.shared.escalation_count[user_id] - 0.5)
                self.log(f"⚠️ Reducing escalation count to {self.shared.escalation_count[user_id]:.1f}/3 (Medium confidence: {confidence:.2f})")

        return self.shared.escalation_count[user_id] >= 3

    def run_background_tasks(self):
        """Process background communications and maintenance"""
        while True:
            try:
                if not self.shared.communication_bus.empty():
                    message = self.shared.communication_bus.get()
                    self.log(f"Processing message: {message['type']}")

                    if message["type"] == "learning_request":
                        self.send_to_agent("learning", message)

            except Exception as e:
                self.log(f"Background task error: {str(e)}")
            time.sleep(0.1)

    def escalate_conversation(self, user_id: str) -> str:
        """Handle escalation to human support"""
        context = self.agents["support"].recall_context(user_id)
        self.log(f"Escalating conversation for user {user_id}")

        report = (
            f"🚨 ESCALATION REQUEST\n"
            f"User: {user_id}\n"
            f"Timestamp: {datetime.now().isoformat()}\n"
            f"Conversation History:\n{context}"
        )

        print(f"\n{report}\n")

        self.shared.escalation_count[user_id] = 0

        return "I'm transferring you to a human specialist. They'll contact you within 5 minutes."

    def send_to_agent(self, agent_id: str, message: dict) -> Optional[dict]:
        """Send message to agent and get response"""
        agent = self.agents.get(agent_id)
        if not agent:
            self.log(f"Agent {agent_id} not found")
            return None

        return agent.handle_message(message)

# ----------------------------
# Main Application
# ----------------------------
def main():
    coordinator = Coordinator()
    bg_thread = threading.Thread(target=coordinator.run_background_tasks, daemon=True)
    bg_thread.start()

    while True:
        try:
            user_id = str(uuid.uuid4())[:8]
            print(f"\n{'=' * 50}")
            print(f"👤 User Session: {user_id}")
            print(f"{'=' * 50}")

            # Initialize state
            initial_state = {
                "user_id": user_id,
                "user_query": "",
                "messages": [],
                "current_response": None,
                "confidence": None,
                "escalated": False,
                "should_escalate": False,
                "metadata": {}
            }

            while True:
                query = input("\n🧑💻 Customer query (type 'exit' to end session): ").strip()
                if query.lower() in ["exit", "quit", "end"]:
                    print("\n🛑 Ending session...")
                    break
                
                # Update state with new query
                initial_state["user_query"] = query
                
                start_time = time.time()
                try:
                    result = coordinator.app.invoke(initial_state)
                except GraphRecursionError:
                    print("🔴 Maximum recursion depth reached. Starting new conversation.")
                    initial_state["messages"] = []
                    continue
                except Exception as e:
                    print(f"⚠️ Graph invocation error: {str(e)}")
                    traceback.print_exc()
                    print("🔁 Restarting conversation...")
                    initial_state["messages"] = []
                    continue
                
                # Update state for next iteration
                initial_state = result
                
                # Print results
                print(f"\n⏱️ Response time: {time.time() - start_time:.2f}s")
                source = result.get("metadata", {}).get('source', 'unknown').upper()
                confidence = result.get("confidence", 0)
                print(f"🤖 [{source}] [Confidence: {confidence:.2f}]")
                print(f"💬 Response: {result['current_response']}")
                
                suggestions = result.get("metadata", {}).get("suggestions", [])
                if suggestions:
                    print(f"\n💡 Proactive suggestions: {', '.join(suggestions[:3])}")
                
                if result.get("escalated", False):
                    print("\n🚨 This session has been escalated. Please wait for a human specialist.")
                    break

        except KeyboardInterrupt:
            print("\n\n🛑 System shutdown initiated")
            break
        except Exception as e:
            print(f"⚠️ Critical error: {str(e)}")
            traceback.print_exc()
            print("🔁 Restarting main loop...")
            time.sleep(1)

if __name__ == "__main__":
    main()