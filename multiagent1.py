import os
import uuid
import chromadb
import re
import json
import time
import random
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, TypedDict
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.pregel import GraphRecursionError
import google.generativeai as genai
from typing import Literal, TypedDict, List, Dict, Optional, Tuple
import schedule

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
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=os.getenv("GEMINI_API_KEY"))

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
        def support_specialist_node(state: SupportState) -> SupportState:
            """Processes user query with knowledge base and LLM"""
            user_id = state["user_id"]
            query = state["user_query"]
            
            # Use SupportAgent to generate response
            response, metadata = self.agents["support"].generate_response(query, user_id)
            
            return {
                "current_response": response,
                "confidence": metadata.get("confidence", 0.5),
                "should_escalate": False,
                "escalated": False,
                "metadata": metadata
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

        workflow.add_node("support_specialist", support_specialist_node)
        workflow.add_node("escalation_check", escalation_check_node)
        workflow.add_node("escalate", escalation_node)
        workflow.add_node("personalize", personalization_node)
        workflow.add_node("update_context", update_context_node)

        workflow.set_entry_point("support_specialist")
        workflow.add_edge("support_specialist", "escalation_check")
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
            print("🔁 Restarting main loop...")
            time.sleep(1)

if __name__ == "__main__":
    main()