#!/usr/bin/env python3
"""
Advanced LMDeploy Agent with Tool Integration and Multi-Modal Support

This advanced agent demonstrates:
- Tool/function calling capabilities
- Multi-modal inference (text + vision)
- Advanced conversation management
- Custom prompt templates
- Performance monitoring
"""

import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime

import lmdeploy
from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, VisionConfig
from lmdeploy.vl import load_image


@dataclass
class AdvancedAgentConfig:
    """Advanced configuration for the LMDeploy Agent."""
    model_path: str
    backend: str = "auto"
    max_batch_size: int = 8
    session_len: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 1024
    log_level: str = "WARNING"
    enable_vision: bool = False
    vision_max_dynamic_patch: int = 6
    tools: List[Dict[str, Any]] = field(default_factory=list)
    custom_templates: Dict[str, str] = field(default_factory=dict)


class ToolRegistry:
    """Registry for managing agent tools/functions."""
    
    def __init__(self):
        self.tools = {}
    
    def register(self, name: str, description: str):
        """Decorator to register a tool function."""
        def decorator(func: Callable):
            self.tools[name] = {
                'function': func,
                'description': description,
                'name': name
            }
            return func
        return decorator
    
    def get_tool(self, name: str) -> Optional[Dict]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all available tools."""
        return list(self.tools.keys())
    
    def get_tools_description(self) -> str:
        """Get formatted description of all tools."""
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool['description']}")
        
        return "Available tools:\n" + "\n".join(descriptions)


class ConversationManager:
    """Manages conversation history with advanced features."""
    
    def __init__(self, max_history: int = 50):
        self.history = []
        self.max_history = max_history
        self.metadata = {}
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.history.append(message)
        
        # Trim history if too long
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self, include_system: bool = True) -> str:
        """Get formatted conversation context."""
        context = ""
        for msg in self.history:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system" and include_system:
                context += f"System: {content}\n\n"
            elif role == "user":
                context += f"Human: {content}\n\n"
            elif role == "assistant":
                context += f"Assistant: {content}\n\n"
        
        return context
    
    def clear(self):
        """Clear conversation history."""
        self.history = []
        self.metadata = {}
    
    def save(self, filepath: str):
        """Save conversation to file."""
        data = {
            "history": self.history,
            "metadata": self.metadata,
            "saved_at": datetime.now().isoformat()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath: str):
        """Load conversation from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.history = data.get("history", [])
        self.metadata = data.get("metadata", {})


class AdvancedLMDeployAgent:
    """
    Advanced agent with tool integration and multi-modal capabilities.
    """
    
    def __init__(self, config: AdvancedAgentConfig):
        self.config = config
        self.pipeline = None
        self.conversation = ConversationManager()
        self.tool_registry = ToolRegistry()
        self.performance_stats = []
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Register default tools
        self._register_default_tools()
        
        # Initialize pipeline
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the LMDeploy pipeline with advanced configuration."""
        self.logger.info(f"Initializing advanced pipeline with model: {self.config.model_path}")
        
        # Configure backend
        if self.config.backend == "pytorch":
            backend_config = PytorchEngineConfig(
                max_batch_size=self.config.max_batch_size,
                session_len=self.config.session_len,
                cache_max_entry_count=0.8,
                block_size=64,
                enable_prefix_caching=True,
                eager_mode=False
            )
        elif self.config.backend == "turbomind":
            backend_config = TurbomindEngineConfig(
                max_batch_size=self.config.max_batch_size,
                session_len=self.config.session_len,
                cache_max_entry_count=0.8,
                cache_block_seq_len=64
            )
        else:
            backend_config = None
        
        # Configure vision if enabled
        vision_config = None
        if self.config.enable_vision:
            vision_config = VisionConfig(
                max_dynamic_patch=self.config.vision_max_dynamic_patch
            )
        
        # Initialize pipeline
        self.pipeline = lmdeploy.pipeline(
            model_path=self.config.model_path,
            backend_config=backend_config,
            log_level=self.config.log_level
        )
        
        self.logger.info("Advanced pipeline initialized successfully!")
    
    def _register_default_tools(self):
        """Register default tools for the agent."""
        
        @self.tool_registry.register("calculate", "Perform mathematical calculations")
        def calculate(expression: str) -> str:
            """Safely evaluate mathematical expressions."""
            try:
                # Simple calculator - only allow basic operations
                allowed_chars = set('0123456789+-*/.() ')
                if not all(c in allowed_chars for c in expression):
                    return "Error: Invalid characters in expression"
                
                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        @self.tool_registry.register("get_time", "Get current date and time")
        def get_time() -> str:
            """Get current timestamp."""
            return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        @self.tool_registry.register("word_count", "Count words in text")
        def word_count(text: str) -> str:
            """Count words in given text."""
            words = len(text.split())
            chars = len(text)
            return f"Word count: {words}, Character count: {chars}"
        
        @self.tool_registry.register("search_conversation", "Search conversation history")
        def search_conversation(query: str) -> str:
            """Search for messages in conversation history."""
            results = []
            for i, msg in enumerate(self.conversation.history):
                if query.lower() in msg["content"].lower():
                    results.append(f"Message {i}: {msg['content'][:100]}...")
            
            if results:
                return "Found messages:\n" + "\n".join(results)
            else:
                return "No matching messages found."
    
    def generate(self, 
                 prompt: Union[str, List[str]], 
                 images: Optional[List] = None,
                 **kwargs) -> Union[str, List[str]]:
        """
        Generate text with optional image inputs.
        
        Args:
            prompt: Text prompt(s)
            images: Optional list of images for multi-modal models
            **kwargs: Generation parameters
            
        Returns:
            Generated response(s)
        """
        start_time = time.time()
        
        # Configure generation
        gen_config = GenerationConfig(
            temperature=kwargs.get('temperature', self.config.temperature),
            top_p=kwargs.get('top_p', self.config.top_p),
            top_k=kwargs.get('top_k', self.config.top_k),
            max_new_tokens=kwargs.get('max_new_tokens', self.config.max_new_tokens),
            do_sample=kwargs.get('do_sample', True),
            ignore_eos=kwargs.get('ignore_eos', False)
        )
        
        # Handle multi-modal input
        if images and self.config.enable_vision:
            # For vision models, combine text and images
            if isinstance(prompt, str):
                inputs = [(prompt, images)]
            else:
                inputs = [(p, images) for p in prompt]
        else:
            inputs = prompt if isinstance(prompt, list) else [prompt]
        
        # Generate response
        responses = self.pipeline(inputs, gen_config=gen_config)
        
        # Extract text from responses
        if isinstance(prompt, str):
            result = responses[0].text
        else:
            result = [resp.text for resp in responses]
        
        # Record performance
        end_time = time.time()
        self.performance_stats.append({
            "timestamp": datetime.now().isoformat(),
            "duration": end_time - start_time,
            "input_type": "multi_modal" if images else "text_only",
            "batch_size": len(prompt) if isinstance(prompt, list) else 1
        })
        
        return result
    
    def chat(self, 
             message: str, 
             images: Optional[List] = None,
             system_prompt: Optional[str] = None,
             use_tools: bool = True) -> str:
        """
        Advanced chat with tool support and multi-modal input.
        
        Args:
            message: User message
            images: Optional images for vision models
            system_prompt: System prompt for context
            use_tools: Whether to enable tool usage
            
        Returns:
            Assistant response
        """
        # Set system prompt if provided
        if system_prompt and not any(msg["role"] == "system" for msg in self.conversation.history):
            self.conversation.add_message("system", system_prompt)
        
        # Add user message
        self.conversation.add_message("user", message, {"has_images": bool(images)})
        
        # Prepare prompt with tools context if enabled
        if use_tools:
            tools_context = self.tool_registry.get_tools_description()
            enhanced_message = f"""You are an AI assistant with access to tools. When you need to use a tool, format your response as:
TOOL_CALL: tool_name(arguments)

{tools_context}

User message: {message}

Please respond naturally and use tools when appropriate."""
        else:
            enhanced_message = message
        
        # Get conversation context
        context = self.conversation.get_context()
        full_prompt = context + f"Human: {enhanced_message}\n\nAssistant:"
        
        # Generate response
        response = self.generate(full_prompt, images=images)
        
        # Process tool calls if present
        if use_tools and "TOOL_CALL:" in response:
            response = self._process_tool_calls(response)
        
        # Add response to conversation
        self.conversation.add_message("assistant", response)
        
        return response
    
    def _process_tool_calls(self, response: str) -> str:
        """Process tool calls in the response."""
        lines = response.split('\n')
        processed_lines = []
        
        for line in lines:
            if line.strip().startswith("TOOL_CALL:"):
                tool_call = line.strip()[10:].strip()  # Remove "TOOL_CALL:" prefix
                
                # Parse tool call (simple parsing for demo)
                if '(' in tool_call and ')' in tool_call:
                    tool_name = tool_call.split('(')[0]
                    args_str = tool_call.split('(')[1].rstrip(')')
                    
                    tool = self.tool_registry.get_tool(tool_name)
                    if tool:
                        try:
                            # Simple argument parsing (for demo purposes)
                            if args_str.strip():
                                result = tool['function'](args_str.strip().strip('"\''))
                            else:
                                result = tool['function']()
                            processed_lines.append(f"Tool result: {result}")
                        except Exception as e:
                            processed_lines.append(f"Tool error: {str(e)}")
                    else:
                        processed_lines.append(f"Unknown tool: {tool_name}")
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def analyze_image(self, image_path: str, question: str = "Describe this image") -> str:
        """
        Analyze an image using vision capabilities.
        
        Args:
            image_path: Path to image file
            question: Question about the image
            
        Returns:
            Image analysis result
        """
        if not self.config.enable_vision:
            return "Vision capabilities not enabled. Set enable_vision=True in config."
        
        try:
            image = load_image(image_path)
            response = self.generate(question, images=[image])
            return response
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def batch_process_with_tools(self, 
                                tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple tasks with tool support.
        
        Args:
            tasks: List of task dictionaries with 'prompt' and optional 'tools'
            
        Returns:
            List of results
        """
        results = []
        
        for i, task in enumerate(tasks):
            prompt = task.get('prompt', '')
            use_tools = task.get('use_tools', True)
            images = task.get('images', None)
            
            start_time = time.time()
            
            if use_tools:
                response = self.chat(prompt, images=images, use_tools=True)
            else:
                response = self.generate(prompt, images=images)
            
            end_time = time.time()
            
            results.append({
                "task_id": i,
                "prompt": prompt,
                "response": response,
                "duration": end_time - start_time,
                "used_tools": use_tools,
                "timestamp": datetime.now().isoformat()
            })
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_stats:
            return {"message": "No performance data available"}
        
        durations = [stat["duration"] for stat in self.performance_stats]
        
        return {
            "total_requests": len(self.performance_stats),
            "average_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_duration": sum(durations),
            "requests_per_minute": len(self.performance_stats) / (sum(durations) / 60) if sum(durations) > 0 else 0
        }
    
    def export_conversation(self, filepath: str):
        """Export conversation history."""
        self.conversation.save(filepath)
        self.logger.info(f"Conversation exported to {filepath}")
    
    def import_conversation(self, filepath: str):
        """Import conversation history."""
        self.conversation.load(filepath)
        self.logger.info(f"Conversation imported from {filepath}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass


def main():
    """Demonstrate advanced agent capabilities."""
    
    # Configuration for advanced features
    config = AdvancedAgentConfig(
        model_path="internlm/internlm3-8b-instruct",  # Change to your model
        backend="auto",
        max_batch_size=4,
        session_len=8192,
        enable_vision=False,  # Set to True if using a vision model
        tools=[],
        custom_templates={}
    )
    
    with AdvancedLMDeployAgent(config) as agent:
        print("=== Advanced LMDeploy Agent Demo ===\n")
        
        # 1. Tool-enhanced conversation
        print("1. Tool-Enhanced Conversation:")
        response1 = agent.chat("What's the current time and calculate 15 * 23?", use_tools=True)
        print(f"Response: {response1}\n")
        
        # 2. Batch processing with tools
        print("2. Batch Processing with Tools:")
        tasks = [
            {"prompt": "Calculate the area of a circle with radius 5", "use_tools": True},
            {"prompt": "Count words in 'Hello world this is a test'", "use_tools": True},
            {"prompt": "What is artificial intelligence?", "use_tools": False}
        ]
        
        batch_results = agent.batch_process_with_tools(tasks)
        for result in batch_results:
            print(f"Task: {result['prompt']}")
            print(f"Response: {result['response'][:150]}...")
            print(f"Duration: {result['duration']:.2f}s\n")
        
        # 3. Conversation management
        print("3. Conversation Management:")
        agent.chat("Remember that my favorite color is blue")
        agent.chat("What's my favorite color?")
        
        # Search conversation
        search_result = agent.tool_registry.get_tool("search_conversation")["function"]("favorite")
        print(f"Search result: {search_result}\n")
        
        # 4. Performance statistics
        print("4. Performance Statistics:")
        stats = agent.get_performance_stats()
        print(f"Performance: {json.dumps(stats, indent=2)}\n")
        
        # 5. Export conversation
        print("5. Export/Import Conversation:")
        agent.export_conversation("conversation_history.json")
        print("Conversation exported successfully\n")
        
        # 6. Vision example (if enabled)
        if config.enable_vision:
            print("6. Vision Analysis:")
            # This would work with a vision-enabled model
            # result = agent.analyze_image("path/to/image.jpg", "What do you see in this image?")
            # print(f"Vision result: {result}")
        else:
            print("6. Vision capabilities disabled (set enable_vision=True to use)\n")
        
        print("=== Advanced Demo Complete ===")


if __name__ == "__main__":
    main()