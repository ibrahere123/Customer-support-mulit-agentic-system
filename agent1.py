import os
import uuid
import chromadb
import re
import json
import schedule
import time
import threading
import random
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List
import google.generativeai as genai
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

class SupportAgent:
    def __init__(self):
        """Initialize autonomous support agent"""
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=".chromadb")
        self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Initialize knowledge base collection
        self.collection = self.client.get_or_create_collection(
            name="support_kb",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize conversation memory
        self.conversation_memory = self.client.get_or_create_collection(
            name="conversation_memory",
            embedding_function=self.embedding_function
        )
        
        # Initialize state trackers
        self.user_contexts = {}
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
        
        # Setup background tasks
        self.optimization_schedule = schedule.Scheduler()
        self.optimization_schedule.every().day.at("03:00").do(self.optimize_knowledge_base)
        self.optimization_schedule.every().day.at("04:00").do(self.self_heal_knowledge_base)  # Added self-healing
        self.maintenance_thread = threading.Thread(target=self.run_scheduled_tasks, daemon=True)
        self.maintenance_thread.start()

    def run_scheduled_tasks(self):
        """Run background optimization tasks"""
        while True:
            self.optimization_schedule.run_pending()
            time.sleep(60)

    def _add_kb_entry(self, question: str, answer: str, source: str = "auto") -> str:
        """Add entry to knowledge base with tracking metadata"""
        entry_id = str(uuid.uuid4())
        self.collection.add(
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

    def _update_kb_entry(self, entry_id: str, success: bool = True):
        """Update entry usage statistics with error handling"""
        try:
            entry = self.collection.get(ids=[entry_id])
            metadata = entry["metadatas"][0]
            
            # Handle missing keys with defaults
            current_uses = metadata.get("uses", 0)
            current_successes = metadata.get("successes", 0)
            
            updates = {
                "uses": int(current_uses) + 1,
                "successes": int(current_successes) + (1 if success else 0),
                "last_used": datetime.now().isoformat()
            }
            
            self.collection.update(
                ids=[entry_id],
                metadatas=[{**metadata, **updates}]
            )
        except Exception as e:
            print(f"Failed to update entry {entry_id}: {str(e)}")

    def query_knowledge(self, raw_query: str) -> Optional[Tuple[str, str, float, str]]:
        """Enhanced similarity search with better validation"""
        query = self.preprocess_query(raw_query)
        
        try:
            # Increase results for better matching
            results = self.collection.query(
                query_texts=[query],
                n_results=5,
                include=["metadatas", "distances", "documents"]
            )
            
            if not results["ids"][0]:
                return None
                
            # Validate top matches with better filtering
            for i in range(len(results["ids"][0])):
                entry_id = results["ids"][0][i]
                distance = results["distances"][0][i]
                answer = results["metadatas"][0][i]["answer"]
                question = results["documents"][0][i]
                confidence = max(0.0, 1 - distance)
                
                # Apply dynamic confidence threshold
                min_confidence = max(0.5, self.confidence_threshold - 0.1)
                if confidence < min_confidence:
                    continue
                    
                # Faster validation with keyword matching
                if self.validate_match(query, question):
                    return answer, entry_id, confidence, question
                    
        except Exception as e:
            print(f"Query error: {str(e)}")
            
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
            print(f"Preprocessing error: {str(e)}")
            return raw_query  # Fallback to original

    def generate_response(self, query: str, user_id: str = "default") -> Tuple[str, dict]:
        # Retrieve conversation context
        context = self.recall_context(user_id)
        
        # Create context-enhanced prompt
        prompt = ""
        if context:
            prompt = f"Conversation History:\n{context}\n\n"
        prompt += f"Current User Query: {query}\n\nResponse:"
        
        # Try knowledge base first
        kb_result = self.query_knowledge(query)
        if kb_result:
            answer, entry_id, confidence, matched_question = kb_result
            self._update_kb_entry(entry_id)
            
            # Refine KB response with LLM
            try:
                refined_answer = self.refine_kb_response(
                    kb_answer=answer,
                    user_query=query,
                    matched_question=matched_question,
                    conversation_history=context
                )
                return refined_answer, {
                    "source": "knowledge_base",
                    "confidence": confidence,
                    "entry_id": entry_id,
                    "matched_question": matched_question,
                    "original_answer": answer
                }
            except Exception as e:
                print(f"KB refinement error: {str(e)}")
                return answer, {
                    "source": "knowledge_base",
                    "confidence": confidence,
                    "entry_id": entry_id,
                    "matched_question": matched_question
                }
        
        # Fallback to Gemini generation with context
        try:
            response = self.model.generate_content(
                f"Continue this customer support conversation:\n{prompt}\n\n"
                "Guidelines:\n"
                "1. Keep responses under 50 words\n"
                "2. Maintain conversational context\n"
                "3. Never invent features or links\n"
                "4. Ask clarifying questions when needed"
            )
            generated_answer = response.text
            
            # Calculate realistic confidence based on relevance
            confidence = self.calculate_response_confidence(query, generated_answer)
            
            # Auto-learn temporary entry
            temp_id = self._add_kb_entry(query, generated_answer, source="pending")
            
            return generated_answer, {
                "source": "generated",
                "confidence": confidence,
                "temp_id": temp_id
            }
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return "Please contact our support team for further assistance.", {
                "source": "error"
            }

    def calculate_response_confidence(self, query: str, response: str) -> float:
        try:
            # Use Gemini to evaluate confidence
            prompt = (
                f"Rate how well this response answers the query (0-100). Consider:\n"
                f"1. Relevance to query\n2. Completeness\n3. Support context\n\n"
                f"QUERY: {query}\nRESPONSE: {response}\n\n"
                f"Output ONLY the numerical score."
            )
            
            score_text = self.model.generate_content(prompt).text
            try:
                # Convert to float and normalize to 0.0-1.0
                score = float(score_text.strip()) / 100
                return max(0.1, min(0.9, score))  # Keep within 10%-90% range
            except:
                return 0.4  # Default if parsing fails
        except Exception as e:
            print(f"Confidence evaluation error: {str(e)}")
            return 0.4  # Fallback value

    def should_escalate(self, confidence: float, user_id: str) -> bool:
        """Improved escalation logic"""
        self.escalation_count[user_id] = self.escalation_count.get(user_id, 0)
        
        # Only escalate after multiple LOW confidence responses
        if confidence < self.escalation_threshold:
            self.escalation_count[user_id] += 1
            print(f"🚨 Escalation count: {self.escalation_count[user_id]}/3 (Confidence: {confidence:.2f} < {self.escalation_threshold})")
        else:
            # Reset counter on clearly successful responses
            if confidence > 0.7:
                self.escalation_count[user_id] = 0
            # Gradually reduce counter on medium confidence
            else:
                self.escalation_count[user_id] = max(0, self.escalation_count[user_id] - 0.5)
    
        return self.escalation_count[user_id] >= 3

    def confirm_answer_quality(self, temp_id: str, was_helpful: bool = True):
        """Finalize auto-learned entries based on real usage"""
        if not was_helpful:
            self.collection.delete(ids=[temp_id])
            return
    
        # Promote temporary entry to permanent
        entry = self.collection.get(ids=[temp_id])
        self.collection.update(
            ids=[temp_id],
            metadatas=[{
                **entry["metadatas"][0],
                "source": "auto_learned",
                "created_at": datetime.now().isoformat()
            }]
        )
    
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
            
            evaluation = self.model.generate_content(evaluation_prompt)
            decision = evaluation.text.strip().upper()
            
            return "YES" in decision
        
        except Exception as e:
            print(f"Self-evaluation error: {str(e)}")
            return False

    def autonomous_learning(self, user_query: str, agent_response: str, response_metadata: dict):
        """Automatically learn from successful interactions"""
        if response_metadata["source"] != "generated":
            return  # Only learn from newly generated responses
        
        # Skip casual questions
        if self.is_casual_question(user_query):
            print("🤖 [Skipped Learning] Casual question detected")
            return
            
        # Self-evaluate response quality
        was_helpful = self.ai_self_evaluate(user_query, agent_response)
        
        # Update knowledge base
        self.confirm_answer_quality(response_metadata["temp_id"], was_helpful)
        
        if was_helpful:
            print("🤖 [AUTO-LEARN] Added new knowledge from successful response")
        else:
            print("🤖 [AUTO-LEARN] Discarded unhelpful response")

    def prune_knowledge_base(self, success_threshold: float = 0.6, max_age_days: int = 90):
        """Automated knowledge base maintenance"""
        all_entries = self.collection.get()
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        entries_to_remove = []
        
        for entry_id, metadata in zip(all_entries["ids"], all_entries["metadatas"]):
            # Calculate success rate
            success_rate = metadata["successes"] / metadata["uses"] if metadata["uses"] > 0 else 0
            
            # Check age
            last_used = datetime.fromisoformat(metadata["last_used"]) if metadata["last_used"] else None
            is_old = last_used and last_used < cutoff_date
            
            if success_rate < success_threshold or is_old:
                entries_to_remove.append(entry_id)
        
        if entries_to_remove:
            self.collection.delete(ids=entries_to_remove)
            print(f"🧹 Pruned {len(entries_to_remove)} knowledge base entries")
            
        return len(entries_to_remove)

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
            if user_id not in self.user_contexts:
                self.user_contexts[user_id] = []
            elif not isinstance(self.user_contexts[user_id], list):
                # Reset if corrupted
                self.user_contexts[user_id] = []
                
            # Add new exchange to context
            self.user_contexts[user_id].append({"role": "user", "content": masked_query})
            self.user_contexts[user_id].append({"role": "assistant", "content": masked_response})
            
            # Check context size and summarize if needed
            context_str = self.recall_context(user_id)
            if len(context_str) > self.context_summary_threshold:
                summarized = self.summarize_context(context_str)
                # Reset context with summary + current exchange
                self.user_contexts[user_id] = [
                    {"role": "system", "content": f"Summary: {summarized}"},
                    {"role": "user", "content": masked_query},
                    {"role": "assistant", "content": masked_response}
                ]
            else:
                # Keep only recent history to manage context length
                if len(self.user_contexts[user_id]) > self.max_history_length * 2:
                    self.user_contexts[user_id] = self.user_contexts[user_id][-self.max_history_length * 2:]
                
            # Add to vector store
            exchange = f"User: {masked_query}\nAgent: {masked_response}"
            self.conversation_memory.add(
                documents=[exchange],
                metadatas=[{"user_id": user_id, "timestamp": datetime.now().isoformat()}],
                ids=[f"{user_id}_{int(time.time())}"]
            )
                
        except Exception as e:
            print(f"Context saving error: {str(e)}")
            # Reset context on error
            self.user_contexts[user_id] = []

    def summarize_context(self, context: str) -> str:
        """Summarize long conversation history using AI"""
        try:
            prompt = (
                f"Create a concise 3-sentence summary of this conversation, "
                f"preserving key user concerns and solutions:\n\n{context}"
            )
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Summarization error: {str(e)}")
            return context[:1500]  # Fallback truncation

    def recall_context(self, user_id: str) -> str:
        """Robust context recall with conversation history"""
        try:
            # Get recent conversation history
            context_messages = self.user_contexts.get(user_id, [])
            
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
            print(f"Context recall error: {str(e)}")
            return ""

    def escalate_conversation(self, user_id: str) -> str:
        """Handle escalation to human support"""
        context = self.recall_context(user_id)
        
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
        self.escalation_count[user_id] = 0
        
        return "I'm transferring you to a human specialist. They'll contact you within 5 minutes."

    def generate_proactive_suggestions(self, user_query: str, response: str) -> List[str]:
        """Generate proactive suggestions based on conversation"""
        try:
            suggestions = []
            
            # Analyze query for key topics
            for topic, keywords in self.proactive_triggers.items():
                if topic in user_query.lower():
                    suggestions.extend(keywords)
            
            # AI-generated suggestions if none found
            if not suggestions:
                prompt = (
                    f"Suggest 2-3 related help topics for this conversation:\n"
                    f"Q: {user_query}\nA: {response}\n"
                    "Output only the suggestions as a comma-separated list."
                )
                ai_suggestions = self.model.generate_content(prompt).text
                suggestions = [s.strip() for s in ai_suggestions.split(",")][:3]
            
            return suggestions
        
        except Exception as e:
            print(f"Suggestion error: {str(e)}")
            return []

    def optimize_knowledge_base(self):
        """Automatically improve knowledge base entries"""
        try:
            print("🔧 Starting knowledge base optimization...")
            entries = self.collection.get()
            optimized_count = 0
            
            for i, (entry_id, metadata) in enumerate(zip(entries["ids"], entries["metadatas"])):
                if metadata.get("uses", 0) > 10:  # Only optimize frequently used entries
                    # AI rewrite for clarity
                    prompt = (
                        f"Improve this support answer for clarity and conciseness:\n"
                        f"{metadata['answer']}\n\n"
                        "Respond ONLY with the improved answer."
                    )
                    improved = self.model.generate_content(prompt).text
                    
                    # Update entry
                    self.collection.update(
                        ids=[entry_id],
                        metadatas=[{**metadata, "answer": improved, "optimized": True}]
                    )
                    optimized_count += 1
            
            print(f"✅ Optimized {optimized_count} knowledge base entries")
            return optimized_count
            
        except Exception as e:
            print(f"Optimization error: {str(e)}")
            return 0

    def self_heal_knowledge_base(self):
        """Identify and repair underperforming knowledge base entries"""
        repaired_count = 0
        all_entries = self.collection.get()
        
        print("\n🛠️ Starting knowledge base self-healing...")
        
        for entry_id, metadata in zip(all_entries["ids"], all_entries["metadatas"]):
            uses = metadata.get("uses", 0)
            if uses < 5:  # Not enough data
                continue
                
            success_rate = metadata.get("successes", 0) / uses
            
            # Flag problematic entries
            if success_rate < 0.4:
                try:
                    # Get full entry data
                    entry = self.collection.get(ids=[entry_id])
                    question = entry["documents"][0]
                    original_answer = metadata["answer"]
                    
                    # Generate improved answer
                    prompt = (
                        f"This knowledge base entry has low success rate ({success_rate:.0%}):\n"
                        f"Q: {question}\nA: {original_answer}\n\n"
                        f"Improve the answer to make it more helpful and accurate:"
                    )
                    improved = self.model.generate_content(prompt).text
                    
                    # Update entry
                    self.collection.update(
                        ids=[entry_id],
                        metadatas=[{
                            **metadata,
                            "answer": improved,
                            "repaired_at": datetime.now().isoformat(),
                            "original_answer": original_answer
                        }]
                    )
                    repaired_count += 1
                    print(f"  - Repaired: {question[:50]}...")
                except Exception as e:
                    print(f"Self-heal failed for {entry_id}: {str(e)}")
        
        print(f"✅ Repaired {repaired_count} knowledge entries")
        return repaired_count

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
                response = self.model.generate_content(
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
            print(f"Sentiment analysis error: {str(e)}")
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
            print(f"Tone adjustment error: {str(e)}")
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
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Refinement error: {str(e)}")
            return kb_answer  # Fallback to original