import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class MessageAction(Enum):
    """Enum to track message modification actions."""
    ADDED = "added"
    EDITED = "edited"
    DELETED = "deleted"

class ChatHistory:
    def __init__(self):
        """
        Initialize the ChatHistory class for managing conversation history.
        This is a standalone class that can be used by other chatbot implementations.
        """
        self.messages: List[Dict[str, Any]] = []
        self.session_id: Optional[str] = None
        self.history_log: List[Dict[str, Any]] = []
        
    def add_message(self, role: str, content: str, **kwargs) -> int:
        """
        Add a message to the conversation history.
        
        Args:
            role (str): The role of the message sender ('user', 'assistant', 'system')
            content (str): The message content
            **kwargs: Additional message attributes
            
        Returns:
            int: The index of the added message
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "id": str(datetime.now().timestamp())  # Simple unique ID
        }
        message.update(kwargs)
        
        self.messages.append(message)
        self._log_action(MessageAction.ADDED, len(self.messages) - 1, message)
        return len(self.messages) - 1
        
    def add_user_message(self, content: str, **kwargs) -> int:
        """Add a user message to the conversation."""
        return self.add_message("user", content, **kwargs)
        
    def add_assistant_message(self, content: str, **kwargs) -> int:
        """Add an assistant message to the conversation."""
        return self.add_message("assistant", content, **kwargs)
        
    def add_system_message(self, content: str, **kwargs) -> int:
        """Add a system message to the conversation."""
        return self.add_message("system", content, **kwargs)
        
    def edit_message(self, index: int, content: str = None, role: str = None, **kwargs) -> bool:
        """
        Edit an existing message in the conversation history.
        
        Args:
            index (int): The index of the message to edit
            content (str, optional): New content for the message
            role (str, optional): New role for the message
            **kwargs: Additional attributes to update
            
        Returns:
            bool: True if successful, False if index is invalid
        """
        if index < 0 or index >= len(self.messages):
            return False
            
        original_message = self.messages[index].copy()
        message = self.messages[index]
        
        if content is not None:
            message["content"] = content
        if role is not None:
            message["role"] = role
        message.update(kwargs)
        message["updated_at"] = datetime.now().isoformat()
        
        self._log_action(MessageAction.EDITED, index, message, original_message)
        return True
        
    def delete_message(self, index: int) -> bool:
        """
        Delete a message from the conversation history.
        
        Args:
            index (int): The index of the message to delete
            
        Returns:
            bool: True if successful, False if index is invalid
        """
        if index < 0 or index >= len(self.messages):
            return False
            
        deleted_message = self.messages.pop(index)
        self._log_action(MessageAction.DELETED, index, deleted_message)
        return True
        
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the conversation history."""
        return self.messages.copy()
        
    def get_message(self, index: int) -> Optional[Dict[str, Any]]:
        """Get a specific message by index."""
        if 0 <= index < len(self.messages):
            return self.messages[index].copy()
        return None
        
    def clear_history(self) -> None:
        """Clear all messages from the conversation history."""
        # Log all messages being cleared
        for i, msg in enumerate(self.messages):
            self._log_action(MessageAction.DELETED, i, msg)
        self.messages.clear()
        
    def set_session_id(self, session_id: str) -> None:
        """Set a session ID for tracking conversations."""
        self.session_id = session_id
        
    def get_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self.session_id
        
    def _log_action(self, action: MessageAction, index: int, new_data: Dict, old_data: Dict = None) -> None:
        """
        Log message modifications for tracking purposes.
        
        Args:
            action (MessageAction): The type of action performed
            index (int): The index of the affected message
            new_data (Dict): The new data for the message
            old_data (Dict, optional): The original data before changes
        """
        log_entry = {
            "action": action.value,
            "index": index,
            "timestamp": datetime.now().isoformat(),
            "new_data": new_data.copy()
        }
        
        if old_data is not None:
            log_entry["old_data"] = old_data.copy()
            
        self.history_log.append(log_entry)
        
    def get_history_log(self) -> List[Dict[str, Any]]:
        """Get the complete history log of modifications."""
        return self.history_log.copy()
        
    def get_message_history(self, index: int) -> Optional[List[Dict[str, Any]]]:
        """
        Get the history of modifications for a specific message.
        
        Args:
            index (int): The index of the message to get history for
            
        Returns:
            List[Dict[str, Any]]: List of modification records for that message
        """
        if index < 0 or index >= len(self.messages):
            return None
            
        return [entry for entry in self.history_log if entry["index"] == index]
        
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
        # Note: history_log is not restored from dict to maintain clean state
        
    def save_to_file(self, filename: str) -> None:
        """Save the chat history to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    def load_from_file(self, filename: str) -> bool:
        """
        Load chat history from a JSON file.
        
        Returns:
            bool: True if successful, False if file not found
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.from_dict(data)
            return True
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with empty history.")
            return False
        except json.JSONDecodeError:
            print(f"Invalid JSON in file {filename}. Starting with empty history.")
            return False
            
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.messages:
            return "No messages in conversation"
            
        summary_parts = []
        for msg in self.messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            # Truncate content to 50 characters for summary
            truncated_content = content[:50] + "..." if len(content) > 50 else content
            summary_parts.append(f"{role}: {truncated_content}")
            
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
        
    def get_message_count(self) -> int:
        """Get the total number of messages."""
        return len(self.messages)
        
    def is_empty(self) -> bool:
        """Check if the conversation history is empty."""
        return len(self.messages) == 0