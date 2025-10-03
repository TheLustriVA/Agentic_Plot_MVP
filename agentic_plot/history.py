import requests
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

class ChatHistory:
    def __init__(self, api_endpoint: str = "http://127.0.0.1:8188"):
        """
        Initialize the ChatHistory with API endpoint.
        
        Args:
            api_endpoint (str): The OpenAI-compatible API endpoint
        """
        self.api_endpoint = api_endpoint.rstrip('/')
        self.messages: List[Dict[str, Any]] = []
        self.session_id: Optional[str] = None
        
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role (str): The role of the message sender ('user', 'assistant', 'system')
            content (str): The message content
            **kwargs: Additional message attributes
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        message.update(kwargs)
        self.messages.append(message)
        
    def add_user_message(self, content: str, **kwargs) -> None:
        """Add a user message to the conversation."""
        self.add_message("user", content, **kwargs)
        
    def add_assistant_message(self, content: str, **kwargs) -> None:
        """Add an assistant message to the conversation."""
        self.add_message("assistant", content, **kwargs)
        
    def add_system_message(self, content: str, **kwargs) -> None:
        """Add a system message to the conversation."""
        self.add_message("system", content, **kwargs)
        
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the conversation history."""
        return self.messages.copy()
        
    def clear_history(self) -> None:
        """Clear all messages from the conversation history."""
        self.messages.clear()
        
    def set_session_id(self, session_id: str) -> None:
        """Set a session ID for tracking conversations."""
        self.session_id = session_id
        
    def get_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self.session_id
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the chat history to a dictionary."""
        return {
            "messages": self.messages,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat()
        }
        
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load chat history from a dictionary."""
        self.messages = data.get("messages", [])
        self.session_id = data.get("session_id")
        
    def save_to_file(self, filename: str) -> None:
        """Save the chat history to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    def load_from_file(self, filename: str) -> None:
        """Load chat history from a JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.from_dict(data)
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with empty history.")
            
    def send_to_api(self, model: str = "Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf", **kwargs) -> Dict[str, Any]:
        """
        Send the current conversation history to the OpenAI-compatible API.
        
        Args:
            model (str): The model to use for the API call
            **kwargs: Additional API parameters
            
        Returns:
            Dict[str, Any]: The API response
        """
        try:
            payload = {
                "model": model,
                "messages": self.messages,
                **kwargs
            }
            
            response = requests.post(
                f"{self.api_endpoint}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {str(e)}")
            
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.messages:
            return "No messages in conversation"
            
        summary_parts = []
        for i, msg in enumerate(self.messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            content = content[:50] + "..." if len(content) > 50 else content
            summary_parts.append(f"{role}: {content}")
            
        return "\n".join(summary_parts)

        
    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """Get the last message in the conversation."""
        return self.messages[-1] if self.messages else None
        
    def get_user_messages(self) -> List[Dict[str, Any]]:
        """Get all user messages."""
        return [msg for msg in self.messages if msg.get('role') == 'user']
        
    def get_assistant_messages(self) -> List[Dict[str, Any]]:
        """Get all assistant messages."""
        return [msg for msg in self.messages if msg.get('role') == 'assistant']

# Example usage:
if __name__ == "__main__":
    # Create chat history instance
    chat = ChatHistory("http://127.0.0.1:8188")
    
    # Add some messages
    chat.add_user_message("Hello, how are you?")
    chat.add_assistant_message("I'm doing well, thank you for asking!")
    chat.add_user_message("Can you help me with Python programming?")
    
    # Display conversation
    print("Current conversation:")
    for msg in chat.get_messages():
        print(f"{msg['role']}: {msg['content']}")
    
    # Save to file
    chat.save_to_file("chat_history.json")
    
    # Clear history
    chat.clear_history()
    print(f"\nAfter clearing: {len(chat.get_messages())} messages")
    
    # Load from file
    chat.load_from_file("chat_history.json")
    print(f"After loading: {len(chat.get_messages())} messages")
