import json
import os

class SystemPromptScheduler:
    def __init__(self, config_file: str):
        """
        Initialize the system prompt scheduler with configuration file.
        
        Args:
            config_file (str): Path to JSON configuration file
        """
        self.config_file = config_file
        self.prompts = []
        self.current_prompt_index = 0
        self.interaction_count = 0
        self.load_config()
        
    def load_config(self):
        """Load system prompts from JSON configuration file."""
        try:
            if not os.path.exists(self.config_file):
                # Create default config file if it doesn't exist
                self.create_default_config()
                return
                
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                
            if 'prompts' in config and isinstance(config['prompts'], list):
                self.prompts = config['prompts']
                # Validate prompt structure
                for i, prompt in enumerate(self.prompts):
                    if not isinstance(prompt, dict):
                        raise ValueError(f"Prompt at index {i} must be a dictionary")
                    if 'content' not in prompt:
                        raise ValueError(f"Prompt at index {i} must have 'content' field")
                    if 'count' not in prompt:
                        raise ValueError(f"Prompt at index {i} must have 'count' field")
            else:
                raise ValueError("Configuration must have 'prompts' key with a list value")
                
        except (json.JSONDecodeError, ValueError, FileNotFoundError) as e:
            print(f"Error loading configuration: {e}")
            print("Creating default configuration...")
            self.create_default_config()
            
    def create_default_config(self):
        """Create a default configuration file."""
        default_prompts = [
            {
                "content": "You are a helpful assistant. Answer questions clearly and concisely.",
                "count": 5
            },
            {
                "content": "You are a technical expert. Provide detailed explanations and code examples when relevant.",
                "count": 10
            },
            {
                "content": "You are a creative writer. Respond with imaginative and engaging content.",
                "count": 15
            }
        ]
        
        config = {"prompts": default_prompts}
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        self.prompts = default_prompts
        print(f"Created default configuration file: {self.config_file}")
        
    def get_current_system_prompt(self) -> str:
        """Get the current system prompt based on interaction count."""
        if not self.prompts:
            return "You are a helpful assistant."
            
        # Find the appropriate prompt based on interaction count
        for i, prompt_config in enumerate(self.prompts):
            if self.interaction_count < prompt_config['count']:
                self.current_prompt_index = i
                return prompt_config['content']
            
        # If we've exceeded all counts, use the last prompt
        self.current_prompt_index = len(self.prompts) - 1
        return self.prompts[-1]['content']
        
    def increment_interaction(self):
        """Increment interaction counter and return True if prompt should change."""
        self.interaction_count += 1
        
        # Check if we need to switch to a new prompt
        if len(self.prompts) > 1:
            current_prompt = self.prompts[self.current_prompt_index]
            if self.interaction_count >= current_prompt['count']:
                # Move to next prompt (wrap around if needed)
                self.current_prompt_index = (self.current_prompt_index + 1) % len(self.prompts)
                return True
        return False
        
    def get_interaction_count(self) -> int:
        """Get current interaction count."""
        return self.interaction_count
        
    def get_current_prompt_info(self) -> dict:
        """Get information about the current prompt."""
        if not self.prompts:
            return {}
            
        current_prompt = self.prompts[self.current_prompt_index]
        return {
            "prompt_index": self.current_prompt_index,
            "prompt_content": current_prompt['content'],
            "interaction_count": self.interaction_count,
            "next_prompt_count": current_prompt['count'] if 'count' in current_prompt else None
        }