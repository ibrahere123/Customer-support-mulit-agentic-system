import schedule
import time
from .agent import SupportAgent

def learning_jobs():
    agent = SupportAgent()
    
    # Daily maintenance
    schedule.every().day.at("02:00").do(
        agent.prune_knowledge_base
    )
    
    # Hourly performance updates
    schedule.every().hour.do(
        agent.auto_learn_from_recent_conversations
    )

    while True:
        schedule.run_pending()
        time.sleep(60)