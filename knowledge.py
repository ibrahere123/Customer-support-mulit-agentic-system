# knowledge_initializer.py (or a similar filename)

import os
import json
import re # Make sure re is imported if you need it for json parsing
import uuid # For generating IDs, though agent._add_kb_entry might handle this

import google.generativeai as genai
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# Assuming your SharedResources class is in a file like 'multi_agent_system.py'
# You would import it like this:
# from multi_agent_system import SharedResources

# For demonstration, let's include a minimal SharedResources class here for context
# In your actual project, this would come from your main agent file.
class SharedResources:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.client = chromadb.PersistentClient(path=".chromadb")
        
        # These will be initialized by the knowledge base function,
        # but declared here for clarity.
        self.knowledge_collection = None 
        self.conversation_collection = None

        # Add other shared resources from your existing SharedResources class
        self.user_contexts = {}
        self.user_profiles = {}
        self.escalation_count = {}
        self.proactive_triggers = {
            "password": ["security", "2fa", "recovery"],
            "order": ["tracking", "cancel", "return"],
            "account": ["verification", "settings", "delete"]
        }
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

# It's better to have a dedicated class or module function for KB operations
# that interacts with SharedResources, rather than creating a dummy SupportAgent.
# Let's create a helper function within this file or a new 'kb_manager.py'
def _add_kb_entry_to_shared_collection(shared_resources: SharedResources, question: str, answer: str, source: str = "expert") -> str:
    """Helper to add an entry to the shared knowledge collection."""
    entry_id = str(uuid.uuid4())
    shared_resources.knowledge_collection.add(
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


def initialize_knowledge_base(shared_resources: SharedResources):
    """
    Initialize with essential support knowledge (append-only) for the multi-agent system.
    This function now takes a SharedResources instance.
    """
    
    # Use the collections from shared_resources
    # Ensure they are initialized in SharedResources' __init__ or before calling this.
    # We'll use the names from your agent's current collection names: support_kb, conversation_memory
    
    # Re-initialize collection binding (ensure it's using the correct names from SharedResources)
    print("Initializing ChromaDB collections...")
    shared_resources.knowledge_collection = shared_resources.client.get_or_create_collection(
        name="support_kb", # This should match the name in SharedResources
        embedding_function=shared_resources.embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
    shared_resources.conversation_collection = shared_resources.client.get_or_create_collection(
        name="conversation_memory", # This should match the name in SharedResources
        embedding_function=shared_resources.embedding_function
    )

    print("Checking existing knowledge base entries...")
    # Get current IDs in the KB to avoid adding duplicates
    existing_kb_ids = set()
    try:
        # Fetch existing documents to prevent re-adding
        existing_docs = shared_resources.knowledge_collection.get(
            ids=shared_resources.knowledge_collection.get()['ids'], # Get all IDs
            include=['documents']
        )
        existing_kb_documents = {doc.lower() for doc in existing_docs.get('documents', [])}
        print(f"Found {len(existing_kb_documents)} existing entries in 'support_kb'.")
    except Exception as e:
        print(f"⚠️ Could not fetch existing KB entries: {e}. Assuming empty for now.")
        existing_kb_documents = set()
    

    # Domain-specific knowledge entries
    # You might want to define these in a separate JSON or YAML file for cleaner management
    security_knowledge = [
        ("How to reset password", "Secure password reset: https://example.com/reset (Never share your password with anyone)"),
        ("Password not working", "Troubleshooting steps: https://example.com/password-help (Contact support if issues persist)"),
        ("Enable two-factor authentication", "Security guide: https://example.com/2fa-setup (Recommended for all accounts)"),
        ("Forgot password", "Password recovery: https://example.com/forgot-password (Enter your email to reset)"),
        ("Change password", "Update password: https://example.com/change-password (Requires current password)"),
    ]
    order_knowledge = [
        ("Where is my order?", "Track your package: https://example.com/tracking (Enter your order number)"),
        ("Cancel order", "Cancellation policy: https://example.com/cancel-order (Must be requested within 24 hours)"),
        ("Return item", "Returns portal: https://example.com/returns (Start return process here)"),
        ("Order status", "Check order status: https://example.com/order-status (View recent orders)"),
        ("Update shipping address","Address changes: https://example.com/update-address (Contact support if order shipped)"),
    ]
    account_knowledge = [
        ("Update account information", "Account settings: https://example.com/account (Edit profile under Settings)"),
        ("Delete account", "Account deletion: https://example.com/delete-account (Irreversible action)"),
        ("Subscription management", "Manage subscriptions: https://example.com/subscriptions (Upgrade/downgrade anytime)"),
        ("Account security settings", "Security dashboard: https://example.com/security (Manage 2FA and devices)"),
        ("Payment methods", "Payment settings: https://example.com/payments (Add/remove payment methods)"),
    ]

    all_knowledge = security_knowledge + order_knowledge + account_knowledge
    new_entries_added_count = 0
    
    print(f"🌱 Attempting to add {len(all_knowledge)} expert entries (append-only, avoiding duplicates).")
    
    for question, answer in all_knowledge:
        # Check if the exact question (document) already exists to prevent re-adding
        if question.lower() in existing_kb_documents:
            # print(f"  Skipping existing entry: '{question}'")
            continue

        _add_kb_entry_to_shared_collection(shared_resources, question, answer, source="expert")
        new_entries_added_count += 1

        # Attempt to generate variations
        try:
            # Added more specific prompt for JSON output and adjusted structure to get lists
            resp = shared_resources.model.generate_content(
                f"Generate 2-3 distinct rephrased ways a customer might ask this specific question:\n"
                f"\"{question}\"\n"
                f"Examples:\n"
                f"- 'How do I do X?' -> ['Steps to do X', 'Guide for X']\n"
                f"- 'Can I reset my password?' -> ['Forgot my password', 'Password help']\n"
                f"Respond ONLY with a JSON array of strings, e.g., [\"variation1\", \"variation2\"]."
            )
            
            # Use regex to robustly extract JSON array from potential conversational text
            json_match = re.search(r'\[.*\]', resp.text, re.DOTALL)
            if json_match:
                variations = json.loads(json_match.group(0))
                if not isinstance(variations, list): # Ensure it's a list
                    raise ValueError("LLM did not return a JSON array.")
            else:
                raise ValueError(f"No JSON array found in LLM response: {resp.text}")

            for variation in variations:
                if isinstance(variation, str) and variation.strip() and variation.lower() not in existing_kb_documents:
                    _add_kb_entry_to_shared_collection(shared_resources, variation, answer, source="expert_variation")
                    new_entries_added_count += 1
                elif not isinstance(variation, str):
                    print(f"  ⚠️ Skipping non-string variation: {variation}")
                elif not variation.strip():
                    print(f"  ⚠️ Skipping empty variation.")
                else:
                    # print(f"  Skipping existing variation: '{variation}'")
                    pass # Already exists, skip
        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON decoding failed for variations of '{question}': {e}. Raw response: {resp.text}")
            # Fallback to manual variations if JSON parsing fails
            manual = {
                "How to reset password": ["Need to reset my password", "Password reset help"],
                "Password not working": ["Can't log in with password", "Password incorrect"],
                "Where is my order?": ["Order tracking", "Status of my order"],
                "Cancel order": ["How to cancel purchase", "Stop my order"],
                "Update account information": ["Change my account details", "Edit profile information"],
            }
            for v in manual.get(question, []):
                if v.lower() not in existing_kb_documents:
                    _add_kb_entry_to_shared_collection(shared_resources, v, answer, source="expert_manual_variation")
                    new_entries_added_count += 1
        except Exception as e:
            print(f"  ⚠️ General error generating variations for '{question}': {e}. Fallback to manual.")
            # Fallback to manual variations for any other error
            manual = {
                "How to reset password": ["Need to reset my password", "Password reset help"],
                "Password not working": ["Can't log in with password", "Password incorrect"],
                "Where is my order?": ["Order tracking", "Status of my order"],
                "Cancel order": ["How to cancel purchase", "Stop my order"],
                "Update account information": ["Change my account details", "Edit profile information"],
            }
            for v in manual.get(question, []):
                if v.lower() not in existing_kb_documents:
                    _add_kb_entry_to_shared_collection(shared_resources, v, answer, source="expert_manual_variation")
                    new_entries_added_count += 1

    print(f"✅ Knowledge base initialization complete. Added {new_entries_added_count} new entries/variations.")