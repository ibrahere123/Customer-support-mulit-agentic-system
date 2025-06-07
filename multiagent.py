import os
import uuid
import chromadb
import re
import json
import schedule
import time
import threading
import random
import queue
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List
import google.generativeai as genai
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Shared Resources
# ----------------------------
class SharedResources:
    """Central repository for shared components across agents"""
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=".chromadb")
        self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
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

    def initialize_knowledge_base(self):
        """Initialize with essential support knowledge (append-only)"""
        # Check if collection exists and has entries
        try:
            count = self.knowledge_collection.count()
            if count > 0:
                print(f"✅ Using existing knowledge base with {count} entries")
                return
        except:
            # Collection might not exist yet
            pass
            
        print("📚 Initializing knowledge base...")
        
        # Domain-specific knowledge entries
        security_knowledge = [
            ("How to reset password", "Secure password reset: https://example.com/reset (Never share your password with anyone)"),
            ("Password not working",   "Troubleshooting steps: https://example.com/password-help (Contact support if issues persist)"),
            ("Enable two-factor authentication", "Security guide: https://example.com/2fa-setup (Recommended for all accounts)"),
            ("Forgot password",        "Password recovery: https://example.com/forgot-password (Enter your email to reset)"),
            ("Change password",        "Update password: https://example.com/change-password (Requires current password)"),
        ]
        order_knowledge = [
            ("Where is my order?",     "Track your package: https://example.com/tracking (Enter your order number)"),
            ("Cancel order",           "Cancellation policy: https://example.com/cancel-order (Must be requested within 24 hours)"),
            ("Return item",            "Returns portal: https://example.com/returns (Start return process here)"),
            ("Order status",           "Check order status: https://example.com/order-status (View recent orders)"),
            ("Update shipping address","Address changes: https://example.com/update-address (Contact support if order shipped)"),
        ]
        account_knowledge = [
            ("Update account information", "Account settings: https://example.com/account (Edit profile under Settings)"),
            ("Delete account",             "Account deletion: https://example.com/delete-account (Irreversible action)"),
            ("Subscription management",    "Manage subscriptions: https://example.com/subscriptions (Upgrade/downgrade anytime)"),
            ("Account security settings",  "Security dashboard: https://example.com/security (Manage 2FA and devices)"),
            ("Payment methods",            "Payment settings: https://example.com/payments (Add/remove payment methods)"),
        ]

        all_knowledge = security_knowledge + order_knowledge + account_knowledge
        print(f"🌱 Adding {len(all_knowledge)} expert entries")
        
        added_count = 0
        for question, answer in all_knowledge:
            # Add main entry
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
            added_count += 1
            
            # Attempt to generate variations
            try:
                resp = self.model.generate_content(
                    f"Generate 2-3 different ways customers might ask this: \"{question}\"\n"
                    "Return a JSON array of strings."
                )
                variations = json.loads(resp.text)
                for variation in variations:
                    variation_id = str(uuid.uuid4())
                    self.knowledge_collection.add(
                        documents=[variation],
                        metadatas=[{
                            "answer": answer,
                            "source": "expert_variation",
                            "uses": 0,
                            "successes": 0,
                            "created_at": datetime.now().isoformat(),
                            "last_used": datetime.now().isoformat(),
                            "original_question": question
                        }],
                        ids=[variation_id]
                    )
                    added_count += 1
            except Exception as e:
                print(f"  ⚠️ Variations failed for '{question}': {e}")
                # Fallback manual variations
                manual = {
                    "How to reset password": ["Need to reset my password", "Password reset help"],
                    "Password not working": ["Can't log in with password", "Password incorrect"],
                    "Where is my order?": ["Order tracking", "Status of my order"],
                    "Cancel order": ["How to cancel purchase", "Stop my order"],
                    "Update account information": ["Change my account details", "Edit profile information"],
                }
                for v in manual.get(question, []):
                    variation_id = str(uuid.uuid4())
                    self.knowledge_collection.add(
                        documents=[v],
                        metadatas=[{
                            "answer": answer,
                            "source": "expert_variation",
                            "uses": 0,
                            "successes": 0,
                            "created_at": datetime.now().isoformat(),
                            "last_used": datetime.now().isoformat(),
                            "original_question": question
                        }],
                        ids=[variation_id]
                    )
                    added_count += 1

        print(f"✅ Knowledge base initialized with {added_count} entries")

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
# Support Agent
# ----------------------------
class SupportAgent(BaseAgent):
    """Handles user interactions and responses"""
    def __init__(self, shared: SharedResources):
        super().__init__("support", shared)
        
    def handle_message(self, message: dict):
        """Process user query and generate response"""
        if message["type"] != "user_query":
            return None
            
        user_id = message["user_id"]
        query = message["content"]
        
        # Generate response
        response, metadata = self.generate_response(query, user_id)
        
        # Remember context
        self.remember_context(user_id, query, response)
        
        # Add proactive suggestions to metadata
        if metadata["source"] in ["knowledge_base", "generated"]:
            suggestions = self.generate_proactive_suggestions(query, response)
            metadata["suggestions"] = suggestions
        
        # Send to coordinator
        return {
            "type": "agent_response",
            "user_id": user_id,
            "response": response,
            "metadata": metadata
        }
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
        # Retrieve conversation context
        context = self.recall_context(user_id)
        
        # Build user profile for personalization
        user_profile = self.build_user_profile(user_id)
        
        # Create context-enhanced prompt
        prompt = ""
        if context:
            prompt = f"Conversation History:\n{context}\n\n"
        prompt += f"Current User Query: {query}\n\nResponse:"
        
        # Try knowledge base first (skip if escalation test query)
        kb_result = None
        if not self.is_escalation_test_query(query):
            kb_result = self.query_knowledge(query)
        
        if kb_result:
            answer, entry_id, confidence, matched_question = kb_result
            self.update_kb_entry(entry_id)
            
            # Refine KB response with LLM
            try:
                refined_answer = self.refine_kb_response(
                    kb_answer=answer,
                    user_query=query,
                    matched_question=matched_question,
                    conversation_history=context
                )
                # Personalize the response
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
                # Personalize even if refinement fails
                personalized_response = self.personalize_response(answer, user_profile)
                return personalized_response, {
                    "source": "knowledge_base",
                    "confidence": confidence,
                    "entry_id": entry_id,
                    "matched_question": matched_question
                }
        
        # Fallback to Gemini generation with context
        try:
            response = self.shared.model.generate_content(
                f"Continue this customer support conversation:\n{prompt}\n\n"
                "Guidelines:\n"
                "1. Keep responses under 50 words\n"
                "2. Maintain conversational context\n"
                "3. Never invent features or links\n"
                "4. Ask clarifying questions when needed"
            )
            generated_answer = response.text
            
            # Personalize the generated response
            personalized_response = self.personalize_response(generated_answer, user_profile)
            
            # Calculate realistic confidence based on relevance
            confidence = self.calculate_response_confidence(query, personalized_response)
            
            # Send learning request to LearningAgent
            self.shared.communication_bus.put({
                "type": "learning_request",
                "user_query": query,
                "agent_response": generated_answer,  # Unpersonalized version
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

    def query_knowledge(self, raw_query: str) -> Optional[Tuple[str, str, float, str]]:
        """Enhanced similarity search with better validation"""
        query = self.preprocess_query(raw_query)
        
        # Crucial: Check if the preprocessed query is empty
        if not query.strip():
            self.log("Preprocessed query is empty, skipping KB lookup.")
            return None

        try:
            # Increase results for better matching
            results = self.shared.knowledge_collection.query(
                query_texts=[query],
                n_results=5,
                include=["metadatas", "distances", "documents"]
            )
            
            if not results or not results["ids"] or not results["ids"][0]: # Added checks for empty results
                return None
                
            # Validate top matches with better filtering
            for i in range(len(results["ids"][0])):
                entry_id = results["ids"][0][i]
                distance = results["distances"][0][i]
                
                # Check for None or empty documents/metadatas
                if not results["metadatas"][0][i] or "answer" not in results["metadatas"][0][i]:
                    self.log(f"Missing metadata/answer for KB entry {entry_id}, skipping.")
                    continue
                if not results["documents"][0][i]:
                    self.log(f"Missing document content for KB entry {entry_id}, skipping.")
                    continue

                answer = results["metadatas"][0][i]["answer"]
                question = results["documents"][0][i]
                confidence = max(0.0, 1 - distance)
                
                # Apply dynamic confidence threshold
                min_confidence = max(0.5, self.shared.confidence_threshold - 0.1)
                if confidence < min_confidence:
                    self.log(f"KB match for '{query}' below min confidence: {confidence:.2f}")
                    continue
                    
                # Faster validation with keyword matching
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
        
        # Require at least 2 significant common keywords
        significant = [kw for kw in common if len(kw) > 3]
        return len(significant) >= 2

    def preprocess_query(self, raw_query: str) -> str:
        """Advanced query normalization and expansion"""
        try:
            # Step 1: Basic cleaning
            query = raw_query.strip().lower()
            
            # Step 2: Fix common typos
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
        
        except Exception as e:
            self.log(f"Preprocessing error: {str(e)}")
            return raw_query  # Fallback to original

    def build_user_profile(self, user_id: str) -> dict:
        """Create personalized user profile from conversation history"""
        # Return cached profile if recently updated
        if user_id in self.shared.user_profiles:
            profile = self.shared.user_profiles[user_id]
            # Ensure interaction_count exists before using modulo
            if profile.get("interaction_count", 0) % self.shared.profile_update_frequency != 0:
                return profile
        
        # Build or update profile
        context = self.recall_context(user_id)
        
        # If no context, return default profile immediately
        if not context.strip():
            # Initialize interaction_count for new profiles
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
            response = self.shared.model.generate_content(prompt)
            
            # Use regex to extract potential JSON object from response
            # This handles cases where LLM might add conversational text around JSON
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                profile_str = json_match.group(0)
                profile = json.loads(profile_str)
            else:
                self.log(f"No JSON found in profile generation response: {response.text}")
                profile = {} # Fallback to empty dict if no JSON found
            
            # Set defaults for missing values
            profile.setdefault("preferred_name", self.shared.default_profile["preferred_name"])
            profile.setdefault("technical_level", self.shared.default_profile["technical_level"])
            profile.setdefault("communication_style", self.shared.default_profile["communication_style"])
            profile.setdefault("known_issues", self.shared.default_profile["known_issues"])
            
            # Track interaction count
            profile["interaction_count"] = self.shared.user_profiles.get(user_id, {}).get("interaction_count", 0) + 1
            
            # Cache the profile
            self.shared.user_profiles[user_id] = profile
            return profile
            
        except json.JSONDecodeError as e:
            self.log(f"JSON decoding error in profile creation: {e}. Raw response: {response.text}")
            return self.shared.default_profile.copy()
        except Exception as e:
            self.log(f"Profile creation general error: {str(e)}")
            return self.shared.default_profile.copy()

    def personalize_response(self, response: str, profile: dict) -> str:
        """Adapt responses to user's preferences"""
        try:
            # 1. Address user by name if available
            if profile["preferred_name"] != "there":
                # Add greeting if response doesn't start with one
                if not re.match(r'^(Hi|Hello|Hey|Good\s(morning|afternoon|evening))', response, re.IGNORECASE):
                    response = f"Hi {profile['preferred_name']}, {response}"
                else:
                    # Replace generic "you" with name
                    response = re.sub(r'\b(?:you|your)\b', profile["preferred_name"], response, flags=re.IGNORECASE)
            
            # 2. Adjust for technical level
            if profile["technical_level"] == "beginner":
                response = self.simplify_technical_terms(response)
            elif profile["technical_level"] == "expert":
                response = self.add_technical_details(response)
            
            # 3. Adjust communication style
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
    def add_technical_details(self, text: str) -> str:
        """Add technical details for expert users"""
        if len(text.split()) < 30:  # Only expand short responses
            prompt = (
                f"Add technical details to this response for an expert user:\n"
                f"{text}\n\n"
                "Add 1-2 technical specifics that an advanced user might find helpful."
            )
            try:
                return self.shared.model.generate_content(prompt).text
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
        if len(text.split()) < 25:  # Only expand short responses
            prompt = (
                f"Add helpful details to this response while keeping it accurate:\n"
                f"{text}\n\n"
                "Add 1-2 useful details that a detail-oriented user might appreciate."
            )
            try:
                return self.shared.model.generate_content(prompt).text
            except:
                pass
        return text

    def calculate_response_confidence(self, query: str, response: str) -> float:
        try:
            # Use Gemini to evaluate confidence
            prompt = (
                f"Rate how well this response answers the query (0-100). Consider:\n"
                f"1. Relevance to query\n2. Completeness\n3. Support context\n\n"
                f"QUERY: {query}\nRESPONSE: {response}\n\n"
                f"Output ONLY the numerical score."
            )
            
            score_text = self.shared.model.generate_content(prompt).text
            try:
                # Convert to float and normalize to 0.0-1.0
                score = float(score_text.strip()) / 100
                return max(0.1, min(0.9, score))  # Keep within 10%-90% range
            except:
                return 0.4  # Default if parsing fails
        except Exception as e:
            self.log(f"Confidence evaluation error: {str(e)}")
            return 0.4  # Fallback value

    def update_kb_entry(self, entry_id: str, success: bool = True):
        """Update entry usage statistics with error handling"""
        try:
            entry = self.shared.knowledge_collection.get(ids=[entry_id])
            metadata = entry["metadatas"][0]
            
            # Handle missing keys with defaults
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
        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Credit card numbers
        text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CARD]', text)
        
        # Phone numbers
        text = re.sub(r'\b(?:\+\d{1,2}\s?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
        
        # API keys/tokens
        text = re.sub(r'\b[a-f0-9]{32}\b', '[TOKEN]', text)
        
        return text

    def remember_context(self, user_id: str, query: str, response: str):
        """Robust context management with summarization and data masking"""
        try:
            # Mask sensitive data first
            masked_query = self.mask_sensitive_data(query)
            masked_response = self.mask_sensitive_data(response)
            
            # Ensure context is properly initialized as a list
            if user_id not in self.shared.user_contexts:
                self.shared.user_contexts[user_id] = []
            elif not isinstance(self.shared.user_contexts[user_id], list):
                # Reset if corrupted
                self.shared.user_contexts[user_id] = []
                
            # Add new exchange to context only if content is not empty
            if masked_query.strip():
                self.shared.user_contexts[user_id].append({"role": "user", "content": masked_query})
            if masked_response.strip():
                self.shared.user_contexts[user_id].append({"role": "assistant", "content": masked_response})
            
            # If no content was added, skip further processing for this exchange
            if not masked_query.strip() and not masked_response.strip():
                self.log(f"Skipped context saving for user {user_id}: both query and response were empty after masking.")
                return

            # Check context size and summarize if needed
            context_str = self.recall_context(user_id)
            if len(context_str) > self.shared.context_summary_threshold:
                summarized = self.summarize_context(context_str)
                # Reset context with summary + current exchange
                self.shared.user_contexts[user_id] = [
                    {"role": "system", "content": f"Summary: {summarized}"},
                    {"role": "user", "content": masked_query},
                    {"role": "assistant", "content": masked_response}
                ]
            else:
                # Keep only recent history to manage context length
                # Ensure it doesn't go below 0 or become empty
                if len(self.shared.user_contexts[user_id]) > self.shared.max_history_length * 2:
                    self.shared.user_contexts[user_id] = self.shared.user_contexts[user_id][-self.shared.max_history_length * 2:]
                    
            # Add to vector store only if exchange is meaningful
            exchange_doc = f"User: {masked_query}\nAgent: {masked_response}".strip()
            if exchange_doc: # Check if the combined document string is not empty
                self.shared.conversation_collection.add(
                    documents=[exchange_doc],
                    metadatas=[{"user_id": user_id, "timestamp": datetime.now().isoformat()}],
                    ids=[f"{user_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"] # Added unique suffix to ID
                )
            else:
                self.log(f"Skipped adding empty exchange to conversation_collection for user {user_id}.")
                    
        except Exception as e:
            self.log(f"Context saving error: {str(e)}")
            # Reset context on error
            self.shared.user_contexts[user_id] = []
    def summarize_context(self, context: str) -> str:
        """Summarize long conversation history using AI"""
        try:
            prompt = (
                f"Create a concise 3-sentence summary of this conversation, "
                f"preserving key user concerns and solutions:\n\n{context}"
            )
            response = self.shared.model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.log(f"Summarization error: {str(e)}")
            return context[:1500]  # Fallback truncation

    def recall_context(self, user_id: str) -> str:
        """Robust context recall with conversation history"""
        try:
            # Get recent conversation history
            context_messages = self.shared.user_contexts.get(user_id, [])
            
            # Ensure we have a list
            if not isinstance(context_messages, list):
                context_messages = []
                
            # Format as string for LLM context
            context_str = "\n".join(
                f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}" 
                for msg in context_messages
            )
            
            return context_str
            
        except Exception as e:
            self.log(f"Context recall error: {str(e)}")
            return ""

    def analyze_sentiment(self, text: str) -> Dict[str, str]:
        """Robust sentiment analysis with error handling"""
        try:
            # Simple keyword-based sentiment analysis as fallback
            negative_words = ["angry", "frustrated", "upset", "disappointed", "furious", "mad", "hate", "terrible"]
            positive_words = ["happy", "pleased", "thank", "great", "wonderful", "excellent", "love", "awesome"]
            
            text_lower = text.lower()
            negative_count = sum(1 for word in negative_words if word in text_lower)
            positive_count = sum(1 for word in positive_words if word in text_lower)
            
            # Use AI for more complex analysis if possible
            try:
                response = self.shared.model.generate_content(
                    f"Analyze the sentiment in this text:\n{text}\n\n"
                    "Respond with JSON format: {{\"sentiment\": \"positive/neutral/negative\", \"intensity\": 1-5}}"
                )
                return json.loads(response.text)
            except:
                # Fallback to keyword analysis
                if negative_count > positive_count:
                    return {"sentiment": "negative", "intensity": min(5, negative_count)}
                elif positive_count > negative_count:
                    return {"sentiment": "positive", "intensity": min(5, positive_count)}
                else:
                    return {"sentiment": "neutral", "intensity": 3}
                
        except Exception as e:
            self.log(f"Sentiment analysis error: {str(e)}")
            return {"sentiment": "neutral", "intensity": 3}

    def adjust_tone(self, response: str, sentiment: Dict[str, str]) -> str:
        """Adjust response tone based on sentiment with fallbacks"""
        try:
            # Handle cases where sentiment dict might be incomplete
            sentiment_type = sentiment.get("sentiment", "neutral")
            intensity = sentiment.get("intensity", 3)
            
            # Try to convert intensity to int if possible
            try:
                intensity = int(intensity)
            except:
                intensity = 3
                
            if sentiment_type == "negative" and intensity >= 4:
                # Simple tone adjustment without API call
                empathetic_phrases = [
                    "I understand this is frustrating.",
                    "I apologize for the inconvenience.",
                    "I can see why you'd be upset about this."
                ]
                return f"{random.choice(empathetic_phrases)} {response}"
            return response
        except Exception as e:
            self.log(f"Tone adjustment error: {str(e)}")
            return response
            
    def refine_kb_response(self, kb_answer: str, user_query: str, 
                        matched_question: str, conversation_history: str) -> str:
        """Refine knowledge base answer to be more conversational"""
        try:
            prompt = (
                f"Rephrase this knowledge base answer to be more natural and conversational. "
                f"Adapt it to the user's specific query while preserving all key information exactly.\n\n"
                f"Rules:\n"
                f"1. Keep links and critical warnings identical\n"
                f"2. Maintain all factual content\n"
                f"3. Add appropriate greeting/closing if needed\n"
                f"4. Respond in 1-2 sentences max\n\n"
                f"User's Actual Query: {user_query}\n"
                f"Matched KB Question: {matched_question}\n"
                f"Conversation History:\n{conversation_history}\n\n"
                f"Original KB Answer: {kb_answer}\n\n"
                f"Refined, Natural Response:"
            )
            
            response = self.shared.model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.log(f"Refinement error: {str(e)}")
            return kb_answer  # Fallback to original

    def generate_proactive_suggestions(self, user_query: str, response: str) -> List[str]:
        """Generate proactive suggestions based on conversation"""
        try:
            suggestions = []
            
            # Analyze query for key topics
            for topic, keywords in self.shared.proactive_triggers.items():
                if topic in user_query.lower():
                    suggestions.extend(keywords)
            
            # AI-generated suggestions if none found
            if not suggestions:
                prompt = (
                    f"Suggest 2-3 related help topics for this conversation:\n"
                    f"Q: {user_query}\nA: {response}\n"
                    "Output only the suggestions as a comma-separated list."
                )
                ai_suggestions = self.shared.model.generate_content(prompt).text
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
        
        # Start scheduler in background thread
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
        # Skip casual questions
        if self.is_casual_question(user_query):
            self.log("Skipped learning: Casual question detected")
            return
            
        # Self-evaluate response quality
        was_helpful = self.ai_self_evaluate(user_query, agent_response)
        
        # Add to knowledge base if helpful
        if was_helpful and confidence > 0.6:
            self.add_kb_entry(user_query, agent_response, "auto_learned")
            self.log(f"Added new knowledge: {user_query[:50]}...")
        else:
            self.log(f"Discarded unhelpful response: {user_query[:50]}...")

    def ai_self_evaluate(self, user_query: str, agent_response: str) -> bool:
        """Use AI to evaluate if response solved the user's problem"""
        try:
            # Advanced AI evaluation
            evaluation_prompt = (
                f"Determine if this response successfully answered the user's query:\n"
                f"USER QUERY: {user_query}\n"
                f"AGENT RESPONSE: {agent_response}\n\n"
                "Answer ONLY 'YES' or 'NO' based on:\n"
                "1. Does the response directly address the query?\n"
                "2. Does it provide a complete solution?\n"
                "3. Would the user likely need to ask follow-up questions?\n"
            )
            
            evaluation = self.shared.model.generate_content(evaluation_prompt)
            decision = evaluation.text.strip().upper()
            
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
            # Calculate success rate
            uses = metadata.get("uses", 1)  # Avoid division by zero
            successes = metadata.get("successes", 0)
            success_rate = successes / uses
            
            # Check age
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
                if metadata.get("uses", 0) > 10:  # Only optimize frequently used entries
                    # AI rewrite for clarity
                    prompt = (
                        f"Improve this support answer for clarity and conciseness:\n"
                        f"{metadata['answer']}\n\n"
                        "Respond ONLY with the improved answer."
                    )
                    improved = self.shared.model.generate_content(prompt).text
                    
                    # Update entry
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
            if uses < 5:  # Not enough data
                continue
                
            success_rate = metadata.get("successes", 0) / uses
            
            # Flag problematic entries
            if success_rate < 0.4:
                try:
                    # Get full entry data
                    entry = self.shared.knowledge_collection.get(ids=[entry_id])
                    question = entry["documents"][0]
                    original_answer = metadata["answer"]
                    
                    # Generate improved answer
                    prompt = (
                        f"This knowledge base entry has low success rate ({success_rate:.0%}):\n"
                        f"Q: {question}\nA: {original_answer}\n\n"
                        f"Improve the answer to make it more helpful and accurate:"
                    )
                    improved = self.shared.model.generate_content(prompt).text
                    
                    # Update entry
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
        self.shared.initialize_knowledge_base()  # Initialize KB
        self.agents = {}
        self.setup_agents()
        self.log("Coordinator initialized")

    def log(self, message: str):
        print(f"[COORDINATOR] {message}")

    def setup_agents(self):
        """Initialize agents"""
        self.agents["support"] = SupportAgent(self.shared)
        self.agents["learning"] = LearningAgent(self.shared)
        self.log("Agents initialized")

    def process_user_query(self, user_id: str, query: str) -> dict:
        """Handle user interaction and return comprehensive result"""
        # Ensure context is initialized for this user
        if user_id not in self.shared.user_contexts:
            self.shared.user_contexts[user_id] = []
            
        # Send to support agent
        agent_response = self.send_to_agent("support", {
            "type": "user_query",
            "user_id": user_id,
            "content": query
        })
        
        if not agent_response:
            return {
                "response": "I'm having trouble processing your request. Please try again later.",
                "metadata": {"source": "error"},
                "escalated": False
            }
        
        response_str = agent_response["response"]
        metadata = agent_response["metadata"]
        
        # Handle escalation
        if self.should_escalate(metadata.get("confidence", 0), user_id):
            escalation_msg = self.escalate_conversation(user_id)
            return {
                "response": escalation_msg,
                "metadata": {"source": "escalation"},
                "escalated": True
            }
            
        return {
            "response": response_str,
            "metadata": metadata,
            "escalated": False
        }
    
    def should_escalate(self, confidence: float, user_id: str) -> bool:
        """Improved escalation logic with detailed logging"""
        # Initialize if needed
        if user_id not in self.shared.escalation_count:
            self.shared.escalation_count[user_id] = 0
            
        # Update escalation count based on confidence
        if confidence < self.shared.escalation_threshold:
            self.shared.escalation_count[user_id] += 1
            self.log(f"🚨 Escalation count increased to {self.shared.escalation_count[user_id]}/3 (Low confidence: {confidence:.2f})")
        else:
            # Reset counter on clearly successful responses
            if confidence > 0.7:
                self.log(f"✅ Resetting escalation counter (High confidence: {confidence:.2f})")
                self.shared.escalation_count[user_id] = 0
            # Gradually reduce counter on medium confidence
            else:
                self.shared.escalation_count[user_id] = max(0, self.shared.escalation_count[user_id] - 0.5)
                self.log(f"⚠️ Reducing escalation count to {self.shared.escalation_count[user_id]:.1f}/3 (Medium confidence: {confidence:.2f})")
    
        return self.shared.escalation_count[user_id] >= 3

    def send_to_agent(self, agent_id: str, message: dict) -> Optional[dict]:
        """Send message to agent and get response"""
        agent = self.agents.get(agent_id)
        if not agent:
            self.log(f"Agent {agent_id} not found")
            return None
            
        return agent.handle_message(message)

    def run_background_tasks(self):
        """Process background communications and maintenance"""
        while True:
            try:
                # Process communication bus messages
                if not self.shared.communication_bus.empty():
                    message = self.shared.communication_bus.get()
                    self.log(f"Processing message: {message['type']}")
                    
                    # Route to appropriate agent
                    if message["type"] == "learning_request":
                        self.send_to_agent("learning", message)
                        
            except Exception as e:
                self.log(f"Background task error: {str(e)}")
            time.sleep(0.1)

    def escalate_conversation(self, user_id: str) -> str:
        """Handle escalation to human support"""
        context = self.agents["support"].recall_context(user_id)
        self.log(f"Escalating conversation for user {user_id}")
        
        # Generate escalation report
        report = (
            f"🚨 ESCALATION REQUEST\n"
            f"User: {user_id}\n"
            f"Timestamp: {datetime.now().isoformat()}\n"
            f"Conversation History:\n{context}"
        )
        
        # In real implementation: Send to ticketing system
        print(f"\n{report}\n")
        
        # Reset escalation count
        self.shared.escalation_count[user_id] = 0
        
        return "I'm transferring you to a human specialist. They'll contact you within 5 minutes."

# ----------------------------
# Main Application
# ----------------------------
def main():
    # Initialize coordinator
    coordinator = Coordinator()
    
    # Start background tasks in a separate thread
    bg_thread = threading.Thread(target=coordinator.run_background_tasks, daemon=True)
    bg_thread.start()
    
    # Main interaction loop
    while True:
        try:
            # Create unique user session
            user_id = str(uuid.uuid4())[:8]
            print(f"\n{'=' * 50}")
            print(f"👤 User Session: {user_id}")
            print(f"{'=' * 50}")
            
            # Reset ALL user-specific state for new session
            coordinator.shared.user_contexts[user_id] = []  # Clear conversation context
            coordinator.shared.user_profiles.pop(user_id, None)  # Remove profile
            coordinator.shared.escalation_count[user_id] = 0  # Reset escalation counter
            
            # Conversation loop
            while True:
                # Get user input
                query = input("\n🧑💻 Customer query (type 'exit' to end session): ").strip()
                if query.lower() in ["exit", "quit", "end"]:
                    print("\n🛑 Ending session...")
                    break
                
                # Process query
                start_time = time.time()
                result = coordinator.process_user_query(user_id, query)
                response_time = time.time() - start_time
                
                response_str = result["response"]
                metadata = result.get("metadata", {})
                
                # Print response
                print(f"\n⏱️ Response time: {response_time:.2f}s")
                source = metadata.get('source', 'unknown').upper()
                confidence = metadata.get('confidence', 0)
                print(f"🤖 [{source}] [Confidence: {confidence:.2f}]")
                
                # Print matched question for KB responses
                if 'matched_question' in metadata:
                    print(f"🔍 Matched Question: {metadata['matched_question']}")
                
                print(f"💬 Response: {response_str}")
                
                # Print proactive suggestions if available
                suggestions = metadata.get("suggestions", [])
                if suggestions:
                    print(f"\n💡 Proactive suggestions: {', '.join(suggestions[:3])}")
                
                # Check for escalation
                if result.get("escalated", False):
                    print("\n🚨 This session has been escalated. Please wait for a human specialist.")
                    break
                    
        except KeyboardInterrupt:
            print("\n\n🛑 System shutdown initiated")
            break
        except Exception as e:
            print(f"⚠️ Critical error: {str(e)}")
            print("🔁 Restarting main loop...")
            time.sleep(1)

if __name__ == "__main__":
    main()