"""
Single-Model Creative Generation System

This module provides a simplified approach testing each model individually
for creative writing tasks, optimized for 32GB VRAM constraint.
"""

import json
import re
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# from .history import ChatHistory


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    file: str
    template: str
    capabilities: List[str]
    vram_estimate: int
    context_size: int = 16384
    description: str = ""


class SingleModelAgent:
    """Agent for testing individual models on creative writing tasks"""
    
    # Model configurations optimized for 32GB VRAM
    MODELS = {
        "qwen3_coder": ModelConfig(
            file="Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf",
            template="qwen2",
            capabilities=["coding", "reasoning", "orchestration", "creative_writing"],
            vram_estimate=28,
            context_size=131072,
            description="30.5B MoE model with 3.3B active params, excellent reasoning and creative abilities"
        ),
        "dark_reasoning": ModelConfig(
            file="L3.1-MOE-4X8B-Dark-Reasoning-Super-Nova-RP-Hermes-R1-Uncensored-25B.Q8_0.gguf",
            template="llama3",
            capabilities=["creative_writing", "psychology", "reasoning", "mature_themes"],
            vram_estimate=18,
            context_size=16384,
            description="25B MOE model optimized for creative writing and complex reasoning"
        ),
        "qwq_planet": ModelConfig(
            file="QWQ-RPMax-Planet-32B-ablated.Q6_K.gguf",
            template="qwen",
            capabilities=["reasoning", "structure", "creative_writing", "coherence"],
            vram_estimate=22,
            context_size=32768,
            description="32B ablated model with strong creative writing performance"
        ),
        "venice": ModelConfig(
            file="venice_q6.gguf",
            template="mistral",
            capabilities=["dialogue", "creative_writing", "characters", "unrestricted"],
            vram_estimate=16,
            context_size=16384,
            description="24B Venice model optimized for dialogue and character work"
        )
    }
    
    def __init__(self, models_path: str = "/home/websinthe/aigen/textgen/text_checkpoints"):
        self.models_path = Path(models_path)
        self.active_server = None
        self.current_model = None
        
    def start_single_model(self, model_name: str, port: int = 8188) -> bool:
        """
        Start a single model server for testing.
        
        Args:
            model_name: Name of the model to start
            port: Port to run the server on
            
        Returns:
            True if server started successfully
        """
        if model_name not in self.MODELS:
            print(f"❌ Unknown model: {model_name}")
            return False
            
        config = self.MODELS[model_name]
        model_path = self.models_path / config.file
        
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return False
        
        # Stop any existing server
        self.stop_server()
        
        # Check VRAM requirement
        if config.vram_estimate > 32:
            print(f"⚠️ Model {model_name} may exceed 32GB VRAM limit ({config.vram_estimate}GB estimated)")
        
        print(f"🚀 Starting {model_name} ({config.description})")
        print(f"   VRAM estimate: {config.vram_estimate}GB")
        print(f"   Context size: {config.context_size}")
        print(f"   Model path: {model_path}")
        
        # Build llama-server command
        cmd = [
            "llama-server",
            "--model", str(model_path),
            "--port", str(port),
            "--ctx-size", str(config.context_size),
            "--n-gpu-layers", "28",
            "--host", "0.0.0.0"
        ]
        
        print(f"🔧 Command: {' '.join(cmd)}")
        
        try:
            # Start the server process
            print("⏳ Starting llama-server...")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Show initial output from server startup
            print("📋 Server startup output:")
            print("-" * 50)
            
            # Read and display initial lines
            startup_lines = []
            timeout_start = time.time()
            
            while time.time() - timeout_start < 30:  # 30 second startup timeout
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    print(f"   {line}")
                    startup_lines.append(line)
                    
                    # Check for common startup indicators
                    if any(indicator in line.lower() for indicator in [
                        "server listening", 
                        "http server listening",
                        "server started",
                        "model loaded",
                        "ready"
                    ]):
                        print("✅ Server appears to be starting successfully")
                        break
                        
                    # Check for common errors
                    if any(error in line.lower() for error in [
                        "error", 
                        "failed", 
                        "cuda error",
                        "out of memory",
                        "unable to load",
                        "exception"
                    ]):
                        print(f"❌ Error detected in startup: {line}")
                        break
                
                # Check if process has died
                if process.poll() is not None:
                    print(f"❌ Process exited with code: {process.poll()}")
                    # Read any remaining output
                    remaining = process.stdout.read()
                    if remaining:
                        print(f"   Final output: {remaining}")
                    return False
                
                time.sleep(0.1)
            
            print("-" * 50)
            
            # Wait for server to be ready with health check
            if self._wait_for_server(port, timeout=60):
                self.active_server = {
                    "process": process,
                    "port": port,
                    "model_name": model_name,
                    "config": config,
                    "startup_output": startup_lines
                }
                self.current_model = model_name
                print(f"✅ {model_name} server ready on port {port}")
                return True
            else:
                print("❌ Server failed health check")
                # Show more output if health check failed
                try:
                    remaining_output = process.stdout.read(1024)  # Read up to 1KB more
                    if remaining_output:
                        print("📋 Additional output:")
                        print(remaining_output)
                except IOError as ioe:
                    print(f" IO error{ioe}")
                
                process.terminate()
                return False
                
        except FileNotFoundError:
            print("❌ llama-server not found in PATH")
            print("   Make sure llama.cpp is installed and llama-server is accessible")
            return False
        except Exception as e:
            print(f"❌ Error starting {model_name} server: {e}")
            return False
    
    def _wait_for_server(self, port: int, timeout: int = 120) -> bool:
        """Wait for server to become ready"""
        start_time = time.time()
        
        print("⏳ Waiting for server to be ready...", end="")
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=2)
                if response.status_code == 200:
                    print(" ✅")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            print(".", end="", flush=True)
            time.sleep(3)
        
        print(" ❌")
        return False
    
    def stop_server(self) -> bool:
        """Stop the active server"""
        if self.active_server:
            try:
                self.active_server["process"].terminate()
                print(f"🛑 Stopped {self.active_server['model_name']} server")
                self.active_server = None
                self.current_model = None
                return True
            except Exception as e:
                print(f"❌ Error stopping server: {e}")
                return False
        return False
    
    def test_creative_writing(self, test_scenarios: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Test the current model on creative writing scenarios.
        
        Args:
            test_scenarios: List of test scenarios with prompts
            
        Returns:
            Dictionary with test results
        """
        if not self.active_server:
            print("❌ No active server")
            return {}
        
        model_name = self.current_model
        config = self.MODELS[model_name]
        port = self.active_server["port"]
        
        print(f"🧪 Testing {model_name} creative writing capabilities")
        
        results = {
            "model_name": model_name,
            "model_config": config,
            "test_results": []
        }
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"  📝 Test {i}/{len(test_scenarios)}: {scenario['name']}")
            
            try:
                response = requests.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "user", "content": scenario["prompt"]}
                        ],
                        "max_tokens": scenario.get("max_tokens", 1000),
                        "temperature": scenario.get("temperature", 0.7)
                    },
                    timeout=240
                )
                
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    word_count = len(content.split())
                    
                    test_result = {
                        "test_name": scenario["name"],
                        "prompt": scenario["prompt"],
                        "response": content,
                        "word_count": word_count,
                        "success": True,
                        "evaluation_notes": scenario.get("evaluation_notes", "")
                    }
                    
                    print(f"    ✅ Generated {word_count} words")
                    
                else:
                    test_result = {
                        "test_name": scenario["name"],
                        "prompt": scenario["prompt"],
                        "response": f"HTTP Error {response.status_code}",
                        "word_count": 0,
                        "success": False,
                        "evaluation_notes": f"Server error: {response.status_code}"
                    }
                    print(f"    ❌ HTTP Error {response.status_code}")
                
            except Exception as e:
                test_result = {
                    "test_name": scenario["name"],
                    "prompt": scenario["prompt"],
                    "response": f"Error: {str(e)}",
                    "word_count": 0,
                    "success": False,
                    "evaluation_notes": f"Exception: {str(e)}"
                }
                print(f"    ❌ Error: {e}")
            
            results["test_results"].append(test_result)
        
        return results
    
    def parse_markdown_input(self, input_path: str) -> Dict[str, Any]:
        """Parse the input markdown file into structured data"""
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Untitled Story"
        
        # Extract sections
        sections = {
            'title': title,
            'setting': self._extract_section(content, "Setting"),
            'characters': self._extract_characters(content),
            'scenario': self._extract_section(content, "Scenario"),
            'endings': self._extract_endings(content)
        }
        
        return sections
    
    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a specific section from markdown content"""
        pattern = rf'## {section_name}\s*\n(.*?)(?=\n## |\n### |\Z)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _extract_characters(self, content: str) -> Dict[str, str]:
        """Extract character descriptions from markdown content"""
        characters = {}
        char_section = self._extract_section(content, "Characters")
        char_pattern = r'### (.+?)\s*\n(.*?)(?=\n### |\n## |\Z)'
        char_matches = re.findall(char_pattern, char_section, re.DOTALL)
        
        for name, description in char_matches:
            characters[name.strip()] = description.strip()
        
        return characters
    
    def _extract_endings(self, content: str) -> Dict[str, str]:
        """Extract potential endings from markdown content"""
        endings = {}
        endings_section = self._extract_section(content, "Potential endings")
        ending_pattern = r'### (.+?)\s*\n(.*?)(?=\n### |\n## |\Z)'
        ending_matches = re.findall(ending_pattern, endings_section, re.DOTALL)
        
        for name, description in ending_matches:
            endings[name.strip()] = description.strip()
        
        return endings
    
    def generate_full_story_output(self, input_path: str, output_dir: str = "outputs") -> Dict[str, str]:
        """
        Generate complete story output using the current model.
        
        Args:
            input_path: Path to input markdown file
            output_dir: Output directory
            
        Returns:
            Dictionary of generated files
        """
        if not self.active_server:
            print("❌ No active server")
            return {}
        
        # Parse input
        story_elements = self.parse_markdown_input(input_path)
        model_name = self.current_model
        
        # Create output directory
        output_path = Path(output_dir) / f"{model_name}_output"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📖 Generating story output using {model_name}")
        print(f"   Story: {story_elements['title']}")
        
        outputs = {}
        
        # Generate plot analysis
        print("🔍 Analyzing plot structure...")
        plot_analysis = self._generate_plot_analysis(story_elements)
        plot_file = output_path / "plot_analysis.md"
        with open(plot_file, 'w', encoding='utf-8') as f:
            f.write(plot_analysis)
        outputs['plot_analysis'] = str(plot_file)
        
        # Generate character descriptions
        print("👥 Expanding character descriptions...")
        characters_content = self._generate_character_descriptions(story_elements)
        char_file = output_path / "characters.md"
        with open(char_file, 'w', encoding='utf-8') as f:
            f.write(characters_content)
        outputs['characters'] = str(char_file)
        
        # Generate world building
        print("🌍 Creating world-building bible...")
        world_content = self._generate_world_building(story_elements)
        world_file = output_path / "world_building.md"
        with open(world_file, 'w', encoding='utf-8') as f:
            f.write(world_content)
        outputs['world_building'] = str(world_file)
        
        # Generate synopses for each ending
        for ending_name in story_elements['endings'].keys():
            print(f"📝 Writing synopsis for: {ending_name}")
            synopsis_content = self._generate_synopsis(story_elements, ending_name)
            synopsis_file = output_path / f"synopsis_{ending_name.lower().replace(' ', '_')}.md"
            with open(synopsis_file, 'w', encoding='utf-8') as f:
                f.write(synopsis_content)
            outputs[f'synopsis_{ending_name}'] = str(synopsis_file)
        
        # Generate performance report
        print("📊 Creating performance report...")
        report_content = self._generate_performance_report(story_elements, outputs)
        report_file = output_path / "performance_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        outputs['performance_report'] = str(report_file)
        
        return outputs
    
    def _generate_with_model(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate content using the current model"""
        if not self.active_server:
            return "Error: No active server"
        
        port = self.active_server["port"]
        model_name = self.current_model
        
        try:
            response = requests.post(
                f"http://localhost:{port}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=240
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"HTTP Error {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _generate_plot_analysis(self, story_elements: Dict) -> str:
        """Generate plot analysis using Christopher Booker's framework"""
        prompt = f"""
        Analyze this story using Christopher Booker's "Seven Basic Plots" framework:
        
        Title: {story_elements['title']}
        Setting: {story_elements['setting']}
        Characters: {json.dumps(story_elements['characters'], indent=2)}
        Scenario: {story_elements['scenario']}
        Potential Endings: {json.dumps(story_elements['endings'], indent=2)}
        
        Please provide:
        1. Which of Booker's seven archetypes best fits this story
        2. How the story maps to the five-stage structure
        3. Analysis of how each ending variant affects the archetype
        4. Detailed plot structure recommendations
        
        Format as a comprehensive markdown analysis.
        """
        
        analysis = self._generate_with_model(prompt, max_tokens=2000)
        
        return f"""# Plot Analysis: {story_elements['title']}

*Generated using {self.current_model}*

{analysis}

---
*Generated by Agentic Plot MVP - Single Model Testing*
"""
    
    def _generate_character_descriptions(self, story_elements: Dict) -> str:
        """Generate expanded character descriptions"""
        characters_md = f"""# Character Descriptions: {story_elements['title']}

*Generated using {self.current_model}*

"""
        
        for char_name, char_desc in story_elements['characters'].items():
            prompt = f"""
            Expand this character description for a story set in: {story_elements['setting']}
            
            Character: {char_name}
            Basic Description: {char_desc}
            Story Scenario: {story_elements['scenario']}
            
            Create a comprehensive character profile including:
            - Physical appearance and distinctive features
            - Personality traits and psychological depth
            - Background and motivations
            - Skills, abilities, and relationships
            - Role in the story and character arc potential
            
            Write 300-500 words in an engaging, descriptive style.
            """
            
            expanded_desc = self._generate_with_model(prompt, max_tokens=800)
            characters_md += f"""
## {char_name}

{expanded_desc}

"""
        
        return characters_md + "\\n---\\n*Generated by Agentic Plot MVP - Single Model Testing*"
    
    def _generate_world_building(self, story_elements: Dict) -> str:
        """Generate world-building bible"""
        prompt = f"""
        Create a comprehensive world-building bible for this story:
        
        Setting: {story_elements['setting']}
        Scenario: {story_elements['scenario']}
        Characters: {list(story_elements['characters'].keys())}
        
        Include detailed sections on:
        1. Geography and physical environment
        2. Society, culture, and politics
        3. History and background lore
        4. Technology, magic, or special systems
        5. Daily life and customs
        6. Key locations relevant to the plot
        
        Write 800-1200 words in a detailed, immersive style that supports the story.
        """
        
        world_building = self._generate_with_model(prompt, max_tokens=2000)
        
        return f"""# World-Building Bible: {story_elements['title']}

*Generated using {self.current_model}*

{world_building}

---
*Generated by Agentic Plot MVP - Single Model Testing*
"""
    
    def _generate_synopsis(self, story_elements: Dict, ending_name: str) -> str:
        """Generate a 1000-word synopsis for a specific ending"""
        ending_desc = story_elements['endings'][ending_name]
        
        prompt = f"""
        Write a compelling 1000-word synopsis for this story:
        
        Title: {story_elements['title']}
        Setting: {story_elements['setting']}
        Main Characters: {json.dumps(story_elements['characters'], indent=2)}
        Central Scenario: {story_elements['scenario']}
        
        Target Ending: {ending_name}
        Ending Description: {ending_desc}
        
        The synopsis should:
        - Follow a clear narrative arc leading to the specified ending
        - Include all major characters and their development
        - Capture the tone and atmosphere of the setting
        - Build tension and conflict naturally
        - Make the ending feel inevitable and satisfying
        - Be exactly 1000 words
        
        Write in an engaging, professional style that would make readers want to experience the full story.
        """
        
        synopsis = self._generate_with_model(prompt, max_tokens=1500, temperature=0.8)
        
        return f"""# Synopsis: {story_elements['title']} - {ending_name}

*Generated using {self.current_model}*

## Target Ending
{ending_desc}

## 1000-Word Synopsis

{synopsis}

---
*Generated by Agentic Plot MVP - Single Model Testing*
"""
    
    def _generate_performance_report(self, story_elements: Dict, outputs: Dict) -> str:
        """Generate a performance report for the model"""
        model_config = self.MODELS[self.current_model]
        
        report = f"""# Performance Report: {self.current_model}

## Model Configuration
- **File**: {model_config.file}
- **Template**: {model_config.template}
- **VRAM Estimate**: {model_config.vram_estimate}GB
- **Context Size**: {model_config.context_size}
- **Capabilities**: {', '.join(model_config.capabilities)}
- **Description**: {model_config.description}

## Story Processed
- **Title**: {story_elements['title']}
- **Characters**: {len(story_elements['characters'])}
- **Endings**: {len(story_elements['endings'])}

## Generated Files
"""
        
        for output_type, file_path in outputs.items():
            if output_type != 'performance_report':
                file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
                report += f"- **{output_type}**: {file_path} ({file_size} bytes)\\n"
        
        report += "\\n## Model Assessment\\n"
        report += "*Assessment to be completed after reviewing generated content*\\n\\n"
        report += "---\\n*Generated by Agentic Plot MVP - Single Model Testing*"
        
        return report
    
    def get_available_models(self) -> Dict[str, ModelConfig]:
        """Get list of available models with their configurations"""
        available = {}
        for name, config in self.MODELS.items():
            model_path = self.models_path / config.file
            if model_path.exists():
                available[name] = config
            else:
                print(f"⚠️ Model file not found: {model_path}")
        return available