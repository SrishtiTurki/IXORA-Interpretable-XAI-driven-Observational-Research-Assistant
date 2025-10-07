# tasks.py - Celery Task Implementation
import asyncio
import os
import sys

# Add the parent directory to Python path so we can import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from celery import Celery

# Use the SAME configuration as main.py
celery_app = Celery(
    'medical_assistant',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
    result_expires=3600,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
)

@celery_app.task(bind=True, name='process_chat_task')
def process_chat_task(self, session_id: str, user_message: str) -> dict:
    """
    Celery Task: Processes the chat response in background
    """
    try:
        # Update task state
        self.update_state(state='PROCESSING', meta={'session_id': session_id})
        
        # Import here to avoid circular imports
        from main import response_generator
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Generate response
            result = loop.run_until_complete(
                response_generator.generate_response(session_id, user_message)
            )
            
            # Add metadata
            result['session_id'] = session_id
            result['task_id'] = self.request.id
            
            return result
            
        finally:
            loop.close()
            
    except Exception as e:
        # Return structured error response
        error_result = {
            "content": f"Error processing your request: {str(e)}",
            "type": "error", 
            "session_id": session_id,
            "task_id": self.request.id if hasattr(self, 'request') else 'unknown',
            "error": True,
            "error_details": str(e)
        }
        return error_result