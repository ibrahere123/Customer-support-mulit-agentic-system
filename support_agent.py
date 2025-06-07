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

# ----------------------------
# Support Agent
# ----------------------------
class SupportAgent:
    """Handles user interactions and responses"""
    def __init__(self, shared):
        super(SupportAgent, self).__init__(self, "support", shared)
        self.log("Support agent initialized")

    def handle_message(self, message: dict):
        """Process incoming messages"""
        if message["type"] == "user_query":
            query = message["query"]
            user_id = message["user_id"]
            
            # Generate response
            response, metadata = self.generate_response(query, user_id)
            
            # Remember context
            self.remember_context(user_id, query, response)
            
            # Personalize response
            profile = self.shared.user_profiles.get(user_id, self.shared.default_profile)
            context = self.recall_context(user_id)
            
            personalized_response = self.shared.communication_bus.put({
                "type": "personalize_response",
                "response": response,
                "profile": profile,
                "context": context
            })
            
            # Proactive suggestions
            suggestions = self.generate_proactive_suggestions(query, response)
            
            return {
                "response": personalized_response,
                "metadata": metadata,
                "suggestions": suggestions
            }
        return None

    def generate_response(self, query: str, user_id: str) -> Tuple[str, dict]:
        """Generate response to user query"""
        # Preprocess the query
        processed_query = self.preprocess_query(query)
        
        # Query knowledge base
        kb_result = self.query_knowledge(processed_query)
        
        if kb_result:
            answer, entry_id, confidence, matched_question = kb_result
            
            # Refine the answer
            refined_answer = self.refine_kb_response(answer, query, processed_query)
            
            # Calculate response confidence
            response_confidence = self.calculate_response_confidence(query, refined_answer)
            
            if response_confidence >= self.shared.confidence_threshold:
                self.log(f"✅ Confident response (score={response_confidence:.2f})")
                self.update_kb_entry(entry_id, success=True)
                return refined_answer, {
                    "source": "knowledge_base",
                    "confidence": response_confidence,
                    "matched_question": matched_question
                }
            else:
                self.log(f"⚠️ Low confidence (score={response_confidence:.2f})")
                self.update_kb_entry(entry_id, success=False)
        else:
            confidence = 0.0
            refined_answer = "I am unable to answer the question."
        
        # Fallback: Generate AI response
        try:
            prompt = f"""
            You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you do not know, do not make things up.
            
            Context: {self.recall_context(user_id)}
            
            Question: {query}
            """
            response = self.shared.model.generate_content(prompt)
            refined_answer = response.text
            confidence = 0.7
        except Exception as e:
            refined_answer = f"Apologies, I encountered an error: {e}"
            confidence = 0.1
        
        return refined_answer, {
            "source": "ai_generated",
            "confidence": confidence
        }

    def is_escalation_test_query(self, query: str) -> bool:
        """Check if the query is a test for escalation"""
        return "escalate" in query.lower()

    def query_knowledge(self, raw_query: str) -> Optional[Tuple[str, str, float, str]]:
        """Enhanced similarity search with better validation"""
        try:
            # Embed the query
            query_embedding = self.shared.embedding_function.embed_documents([raw_query])[0]
            
            # Perform similarity search
            results = self.shared.knowledge_collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                where={}
            )
            
            if not results or not results['ids'] or not results['metadatas']:
                return None
            
            # Validate the match
            answer = results['metadatas'][0][0]['answer']
            entry_id = results['ids'][0][0]
            kb_question = results['documents'][0][0]
            
            if not self.validate_match(raw_query, kb_question):
                return None
            
            # Calculate confidence (higher is better)
            confidence = float(results['distances'][0][0])
            
            self.log(f"✅ Knowledge base match (id={entry_id}, score={confidence:.2f})")
            return answer, entry_id, confidence, kb_question
        except Exception as e:
            print(f"  ⚠️ Knowledge base query failed: {e}")
            return None

    def validate_match(self, query: str, kb_question: str) -> bool:
        """Validate that the knowledge base question matches the user query"""
        # Normalize strings
        query = query.lower().strip()
        kb_question = kb_question.lower().strip()
        
        # Check for direct match
        if query == kb_question:
            return True
        
        # Check if query contains keywords from the question
        keywords = re.findall(r'\b(\w+)\b', kb_question)
        if all(keyword in query for keyword in keywords):
            return True
        
        return False

    def preprocess_query(self, raw_query: str) -> str:
        """Advanced query normalization and expansion"""
        try:
            # Remove special characters and extra spaces
            query = re.sub(r'[^\w\s]', '', raw_query).lower().strip()
            
            # Expand common abbreviations
            query = re.sub(r"\b(im|i'm)\b", "i am", query)
            query = re.sub(r"\b(u|you)\b", "you", query)
            query = re.sub(r"\b(ur|your)\b", "your", query)
            query = re.sub(r"\b(pls|please)\b", "please", query)
            
            # Correct common typos
            query = re.sub(r"\b(passwrd|pasword)\b", "password", query)
            
            # Remove stop words (very basic list)
            stop_words = ['the', 'a', 'an', 'is', 'are', 'was', 'were']
            query = ' '.join([word for word in query.split() if word not in stop_words])
            
            return query
        except Exception as e:
            print(f"  ⚠️ Query preprocessing failed: {e}")
            return raw_query

    def calculate_response_confidence(self, query: str, response: str) -> float:
        """Calculate confidence score using AI"""
        try:
            prompt = (
                f"Rate the confidence (0.0-1.0) that this response answers the query:\n"
                f"Query: {query}\n"
                f"Response: {response}\n"
                f"Respond ONLY with the confidence score (e.g. '0.8')."
            )
            
            # Attempt to generate content and parse the confidence score
            ai_response = self.shared.model.generate_content(prompt)
            confidence = float(ai_response.text)
            
            # Basic validation
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("Confidence score out of range")
            
            return confidence
        except Exception as e:
            print(f"  ⚠️ Confidence calculation failed: {e}")
            return 0.5  # Fallback

    def update_kb_entry(self, entry_id: str, success: bool = True):
        """Update entry usage statistics with error handling"""
        try:
            results = self.shared.knowledge_collection.get(ids=[entry_id])
            if not results or not results['metadatas']:
                print(f"  ⚠️ Entry not found: {entry_id}")
                return
            
            metadata = results['metadatas'][0]
            metadata['uses'] += 1
            if success:
                metadata['successes'] += 1
            metadata['last_used'] = datetime.now().isoformat()
            
            self.shared.knowledge_collection.update(
                ids=[entry_id],
                metadatas=[metadata]
            )
            self.log(f"✅ Updated KB entry (id={entry_id}, success={success})")
        except Exception as e:
            print(f"  ⚠️ KB update failed: {e}")

    def mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive information in the text"""
        # Replace email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[email protected]', text)
        
        # Replace phone numbers
        text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[phone number]', text)
        
        # Replace credit card numbers
        text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[credit card number]', text)
        
        return text

    def remember_context(self, user_id: str, query: str, response: str):
        """Robust context management with summarization and data masking"""
        try:
            # Mask sensitive data
            masked_query = self.mask_sensitive_data(query)
            masked_response = self.mask_sensitive_data(response)
            
            # Create context entry
            entry = {
                "user_id": user_id,
                "role": "user",
                "content": masked_query,
                "timestamp": datetime.now().isoformat()
            }
            self.shared.conversation_collection.add(
                documents=[masked_query],
                metadatas=[entry],
                ids=[str(uuid.uuid4())]
            )
            
            entry = {
                "user_id": user_id,
                "role": "agent",
                "content": masked_response,
                "timestamp": datetime.now().isoformat()
            }
            self.shared.conversation_collection.add(
                documents=[masked_response],
                metadatas=[entry],
                ids=[str(uuid.uuid4())]
            )
            
            # Summarize if too long
            context = self.recall_context(user_id)
            if len(context) > self.shared.context_summary_threshold:
                summary = self.summarize_context(context)
                
                # Replace old context with summary
                self.shared.conversation_collection.delete(where={"user_id": user_id})
                
                entry = {
                    "user_id": user_id,
                    "role": "system",
                    "content": summary,
                    "timestamp": datetime.now().isoformat()
                }
                self.shared.conversation_collection.add(
                    documents=[summary],
                    metadatas=[entry],
                    ids=[str(uuid.uuid4())]
                )
            
            self.log("✅ Context updated")
        except Exception as e:
            print(f"  ⚠️ Context management failed: {e}")

    def summarize_context(self, context: str) -> str:
        """Summarize long conversation history using AI"""
        try:
            prompt = f"""
            Summarize this conversation history:
            {context}
            
            Focus on key details and user preferences.
            Respond ONLY with the summary.
            """
            response = self.shared.model.generate_content(prompt)
            return response.text
        except:
            return context  # Fallback: return original

    def recall_context(self, user_id: str) -> str:
        """Robust context recall with conversation history"""
        try:
            results = self.shared.conversation_collection.query(
                query_texts=[user_id],
                n_results=self.shared.max_history_length,
                where={"user_id": user_id}
            )
            
            # Sort by timestamp (assuming 'timestamp' is stored in metadata)
            history = sorted(results['metadatas'][0], key=lambda x: x.get('timestamp', ''))
            
            # Format the history
            context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in history])
            
            return context
        except Exception as e:
            print(f"  ⚠️ Context recall failed: {e}")
            return ""

    def refine_kb_response(self, kb_answer: str, user_query: str, processed_query: str) -> str:
        """Refine knowledge base answer to be more conversational"""
        try:
            prompt = f"""
            Refine this knowledge base answer to be more conversational and tailored to the user's query:
            User query: {user_query}
            Processed query: {processed_query}
            Knowledge base answer: {kb_answer}
            
            Respond ONLY with the refined answer.
            """
            response = self.shared.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"  ⚠️ KB refinement failed: {e}")
            return kb_answer