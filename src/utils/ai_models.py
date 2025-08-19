"""
AI Model Management and Utilities

This module provides centralized AI model management:
- OpenAI GPT model integration
- Anthropic Claude model support
- Model switching and fallback logic
- Performance monitoring and caching
- Token usage tracking and optimization

Ensures consistent AI performance across all automation modules.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import openai
import anthropic
from functools import wraps
import hashlib

# Performance monitoring
from collections import defaultdict, deque


class ModelProvider(Enum):
    """AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class ModelType(Enum):
    """Types of AI models"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"


@dataclass
class ModelConfig:
    """AI model configuration"""
    
    provider: ModelProvider
    model_name: str
    model_type: ModelType
    
    # Parameters
    temperature: float = 0.3
    max_tokens: int = 4000
    top_p: float = 1.0
    
    # Performance settings
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Cost and usage
    cost_per_token: float = 0.0
    max_tokens_per_minute: int = 10000


@dataclass
class ModelResponse:
    """Standardized model response"""
    
    content: str
    provider: ModelProvider
    model: str
    
    # Metadata
    tokens_used: int = 0
    cost: float = 0.0
    response_time: float = 0.0
    
    # Quality metrics
    confidence: float = 0.0
    finish_reason: str = "completed"
    
    # Raw response (for debugging)
    raw_response: Dict = None


@dataclass
class UsageStats:
    """Model usage statistics"""
    
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    
    # Error tracking
    error_count: int = 0
    success_rate: float = 100.0
    
    # Performance metrics
    requests_per_minute: List[int] = None
    response_times: List[float] = None


class AIModelManager:
    """
    Centralized AI model management system
    
    Features:
    - Multi-provider support (OpenAI, Anthropic)
    - Automatic failover and retry logic
    - Usage tracking and cost monitoring
    - Response caching for efficiency
    - Performance optimization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = config or self._load_default_config()
        
        # Initialize clients
        self.openai_client = None
        self.anthropic_client = None
        self._initialize_clients()
        
        # Model configurations
        self.models = self._load_model_configs()
        
        # Performance tracking
        self.usage_stats = defaultdict(UsageStats)
        self.response_cache = {}
        self.cache_ttl = 3600  # 1 hour default
        
        # Rate limiting
        self.request_history = defaultdict(deque)
        self.rate_limits = self._load_rate_limits()
        
        self.logger.info("AI Model Manager initialized")
    
    async def generate_text(self, 
                           prompt: str,
                           model_name: str = "gpt-4",
                           **kwargs) -> ModelResponse:
        """
        Generate text using specified AI model
        
        Args:
            prompt: Input text prompt
            model_name: Model to use for generation
            **kwargs: Additional model parameters
            
        Returns:
            ModelResponse with generated text and metadata
        """
        
        start_time = time.time()
        
        # Get model configuration
        model_config = self.models.get(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not configured")
        
        # Check rate limits
        await self._check_rate_limits(model_name)
        
        # Check cache
        cache_key = self._generate_cache_key(prompt, model_name, kwargs)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            self.logger.debug(f"Using cached response for {model_name}")
            return cached_response
        
        # Generate response with retry logic
        response = await self._generate_with_retry(prompt, model_config, **kwargs)
        
        # Calculate metrics
        response_time = time.time() - start_time
        response.response_time = response_time
        
        # Update usage statistics
        self._update_usage_stats(model_name, response)
        
        # Cache response
        self._cache_response(cache_key, response)
        
        return response
    
    async def generate_chat_response(self,
                                   messages: List[Dict[str, str]],
                                   model_name: str = "gpt-4",
                                   **kwargs) -> ModelResponse:
        """
        Generate chat response using conversation history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model_name: Model to use for response
            **kwargs: Additional model parameters
            
        Returns:
            ModelResponse with chat response
        """
        
        model_config = self.models.get(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not configured")
        
        if model_config.provider == ModelProvider.OPENAI:
            return await self._openai_chat_completion(messages, model_config, **kwargs)
        elif model_config.provider == ModelProvider.ANTHROPIC:
            return await self._anthropic_chat_completion(messages, model_config, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
    
    async def classify_text(self,
                           text: str,
                           categories: List[str],
                           model_name: str = "gpt-3.5-turbo",
                           **kwargs) -> ModelResponse:
        """
        Classify text into predefined categories
        
        Args:
            text: Text to classify
            categories: List of possible categories
            model_name: Model to use for classification
            **kwargs: Additional parameters
            
        Returns:
            ModelResponse with classification result
        """
        
        classification_prompt = f"""
Classify the following text into one of these categories: {', '.join(categories)}

Text: {text}

Response format: Return only the category name, nothing else.
"""
        
        return await self.generate_text(classification_prompt, model_name, **kwargs)
    
    async def extract_insights(self,
                             data: str,
                             context: str = "",
                             model_name: str = "gpt-4",
                             **kwargs) -> ModelResponse:
        """
        Extract insights from data using AI
        
        Args:
            data: Data to analyze
            context: Additional context for analysis
            model_name: Model to use
            **kwargs: Additional parameters
            
        Returns:
            ModelResponse with extracted insights
        """
        
        insights_prompt = f"""
Analyze the following data and extract key insights:

Context: {context}

Data:
{data}

Please provide:
1. Key patterns and trends
2. Notable statistics
3. Business implications
4. Actionable recommendations

Format as a structured analysis with clear sections.
"""
        
        return await self.generate_text(insights_prompt, model_name, **kwargs)
    
    async def summarize_content(self,
                               content: str,
                               max_length: int = 200,
                               model_name: str = "gpt-3.5-turbo",
                               **kwargs) -> ModelResponse:
        """
        Summarize long content into concise summary
        
        Args:
            content: Content to summarize
            max_length: Maximum summary length in words
            model_name: Model to use
            **kwargs: Additional parameters
            
        Returns:
            ModelResponse with summary
        """
        
        summary_prompt = f"""
Summarize the following content in no more than {max_length} words:

{content}

Focus on the most important points and key takeaways.
"""
        
        return await self.generate_text(summary_prompt, model_name, max_tokens=max_length*2, **kwargs)
    
    def get_usage_stats(self, model_name: str = None) -> Dict[str, UsageStats]:
        """Get usage statistics for models"""
        
        if model_name:
            return {model_name: self.usage_stats.get(model_name, UsageStats())}
        return dict(self.usage_stats)
    
    def clear_cache(self, model_name: str = None) -> None:
        """Clear response cache"""
        
        if model_name:
            # Clear cache for specific model
            keys_to_remove = [k for k in self.response_cache.keys() if model_name in k]
            for key in keys_to_remove:
                del self.response_cache[key]
        else:
            # Clear all cache
            self.response_cache.clear()
        
        self.logger.info(f"Cache cleared for {model_name or 'all models'}")
    
    async def _generate_with_retry(self,
                                  prompt: str,
                                  model_config: ModelConfig,
                                  **kwargs) -> ModelResponse:
        """Generate response with retry logic"""
        
        last_exception = None
        
        for attempt in range(model_config.retry_attempts):
            try:
                if model_config.provider == ModelProvider.OPENAI:
                    return await self._openai_completion(prompt, model_config, **kwargs)
                elif model_config.provider == ModelProvider.ANTHROPIC:
                    return await self._anthropic_completion(prompt, model_config, **kwargs)
                else:
                    raise ValueError(f"Unsupported provider: {model_config.provider}")
                    
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed for {model_config.model_name}: {e}")
                
                if attempt < model_config.retry_attempts - 1:
                    await asyncio.sleep(model_config.retry_delay * (2 ** attempt))  # Exponential backoff
        
        # All attempts failed
        self.logger.error(f"All attempts failed for {model_config.model_name}")
        raise last_exception
    
    async def _openai_completion(self,
                               prompt: str,
                               model_config: ModelConfig,
                               **kwargs) -> ModelResponse:
        """Generate OpenAI completion"""
        
        # Merge configuration with runtime parameters
        params = {
            "model": model_config.model_name,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", model_config.temperature),
            "max_tokens": kwargs.get("max_tokens", model_config.max_tokens),
            "top_p": kwargs.get("top_p", model_config.top_p)
        }
        
        try:
            response = await openai.Completion.acreate(**params)
            
            # Extract response data
            content = response.choices[0].text.strip()
            tokens_used = response.usage.total_tokens
            
            return ModelResponse(
                content=content,
                provider=ModelProvider.OPENAI,
                model=model_config.model_name,
                tokens_used=tokens_used,
                cost=self._calculate_cost(tokens_used, model_config),
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI completion failed: {e}")
            raise
    
    async def _openai_chat_completion(self,
                                    messages: List[Dict[str, str]],
                                    model_config: ModelConfig,
                                    **kwargs) -> ModelResponse:
        """Generate OpenAI chat completion"""
        
        params = {
            "model": model_config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", model_config.temperature),
            "max_tokens": kwargs.get("max_tokens", model_config.max_tokens)
        }
        
        try:
            response = await openai.ChatCompletion.acreate(**params)
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            return ModelResponse(
                content=content,
                provider=ModelProvider.OPENAI,
                model=model_config.model_name,
                tokens_used=tokens_used,
                cost=self._calculate_cost(tokens_used, model_config),
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI chat completion failed: {e}")
            raise
    
    async def _anthropic_completion(self,
                                  prompt: str,
                                  model_config: ModelConfig,
                                  **kwargs) -> ModelResponse:
        """Generate Anthropic completion"""
        
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
        
        try:
            response = await self.anthropic_client.completions.create(
                model=model_config.model_name,
                prompt=f"Human: {prompt}\n\nAssistant:",
                max_tokens_to_sample=kwargs.get("max_tokens", model_config.max_tokens),
                temperature=kwargs.get("temperature", model_config.temperature)
            )
            
            content = response.completion.strip()
            
            # Estimate tokens (Anthropic doesn't always provide exact count)
            tokens_used = len(content.split()) * 1.3  # Rough estimation
            
            return ModelResponse(
                content=content,
                provider=ModelProvider.ANTHROPIC,
                model=model_config.model_name,
                tokens_used=int(tokens_used),
                cost=self._calculate_cost(tokens_used, model_config),
                raw_response=response
            )
            
        except Exception as e:
            self.logger.error(f"Anthropic completion failed: {e}")
            raise
    
    async def _anthropic_chat_completion(self,
                                       messages: List[Dict[str, str]],
                                       model_config: ModelConfig,
                                       **kwargs) -> ModelResponse:
        """Generate Anthropic chat completion"""
        
        # Convert messages to Anthropic format
        conversation = ""
        for msg in messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            conversation += f"{role}: {msg['content']}\n\n"
        
        conversation += "Assistant:"
        
        return await self._anthropic_completion(conversation, model_config, **kwargs)
    
    def _initialize_clients(self) -> None:
        """Initialize AI model clients"""
        
        # Initialize OpenAI
        if "openai" in self.config:
            openai.api_key = self.config["openai"]["api_key"]
            self.openai_client = openai
            self.logger.info("OpenAI client initialized")
        
        # Initialize Anthropic
        if "anthropic" in self.config:
            self.anthropic_client = anthropic.AsyncAnthropic(
                api_key=self.config["anthropic"]["api_key"]
            )
            self.logger.info("Anthropic client initialized")
    
    def _load_model_configs(self) -> Dict[str, ModelConfig]:
        """Load model configurations"""
        
        models = {}
        
        # OpenAI models
        if "openai" in self.config:
            openai_config = self.config["openai"]
            
            models["gpt-4"] = ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                model_type=ModelType.CHAT,
                temperature=openai_config.get("temperature", 0.3),
                max_tokens=openai_config.get("max_tokens", 4000),
                cost_per_token=0.00003  # Approximate cost
            )
            
            models["gpt-3.5-turbo"] = ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                model_type=ModelType.CHAT,
                temperature=openai_config.get("temperature", 0.3),
                max_tokens=openai_config.get("max_tokens", 4000),
                cost_per_token=0.000002  # Approximate cost
            )
        
        # Anthropic models
        if "anthropic" in self.config:
            anthropic_config = self.config["anthropic"]
            
            models["claude-3-opus"] = ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-opus-20240229",
                model_type=ModelType.CHAT,
                temperature=anthropic_config.get("temperature", 0.3),
                max_tokens=anthropic_config.get("max_tokens", 4000),
                cost_per_token=0.000015  # Approximate cost
            )
        
        return models
    
    def _load_rate_limits(self) -> Dict[str, int]:
        """Load rate limiting configuration"""
        
        return {
            "gpt-4": 10000,  # tokens per minute
            "gpt-3.5-turbo": 40000,
            "claude-3-opus": 10000
        }
    
    async def _check_rate_limits(self, model_name: str) -> None:
        """Check and enforce rate limits"""
        
        current_time = time.time()
        rate_limit = self.rate_limits.get(model_name, 10000)
        
        # Clean old entries (older than 1 minute)
        history = self.request_history[model_name]
        while history and current_time - history[0] > 60:
            history.popleft()
        
        # Check current usage
        current_usage = len(history)
        if current_usage >= rate_limit:
            wait_time = 60 - (current_time - history[0])
            self.logger.warning(f"Rate limit hit for {model_name}, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        # Record this request
        history.append(current_time)
    
    def _generate_cache_key(self, prompt: str, model_name: str, kwargs: Dict) -> str:
        """Generate cache key for response caching"""
        
        # Create deterministic cache key
        cache_data = {
            "prompt": prompt,
            "model": model_name,
            "params": sorted(kwargs.items())
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[ModelResponse]:
        """Get cached response if available and valid"""
        
        if cache_key not in self.response_cache:
            return None
        
        cached_data, timestamp = self.response_cache[cache_key]
        
        # Check if cache is still valid
        if time.time() - timestamp > self.cache_ttl:
            del self.response_cache[cache_key]
            return None
        
        return cached_data
    
    def _cache_response(self, cache_key: str, response: ModelResponse) -> None:
        """Cache response for future use"""
        
        self.response_cache[cache_key] = (response, time.time())
        
        # Limit cache size (simple LRU-like behavior)
        if len(self.response_cache) > 1000:
            # Remove oldest 10% of entries
            sorted_items = sorted(self.response_cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_items[:100]:
                del self.response_cache[key]
    
    def _calculate_cost(self, tokens_used: int, model_config: ModelConfig) -> float:
        """Calculate cost based on token usage"""
        
        return tokens_used * model_config.cost_per_token
    
    def _update_usage_stats(self, model_name: str, response: ModelResponse) -> None:
        """Update usage statistics"""
        
        stats = self.usage_stats[model_name]
        
        stats.total_requests += 1
        stats.total_tokens += response.tokens_used
        stats.total_cost += response.cost
        
        # Update average response time
        total_time = stats.average_response_time * (stats.total_requests - 1) + response.response_time
        stats.average_response_time = total_time / stats.total_requests
        
        # Update success rate
        if response.finish_reason in ["completed", "stop"]:
            stats.success_rate = (stats.success_rate * (stats.total_requests - 1) + 100) / stats.total_requests
        else:
            stats.error_count += 1
            stats.success_rate = (stats.success_rate * (stats.total_requests - 1)) / stats.total_requests
        
        # Track recent performance
        if stats.response_times is None:
            stats.response_times = deque(maxlen=100)
        stats.response_times.append(response.response_time)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        
        return {
            "openai": {
                "api_key": "your-openai-api-key",
                "model": "gpt-4",
                "temperature": 0.3,
                "max_tokens": 4000
            },
            "anthropic": {
                "api_key": "your-anthropic-api-key",
                "model": "claude-3-opus-20240229",
                "max_tokens": 4000
            }
        }


# Utility decorators
def rate_limit(calls_per_minute: int = 60):
    """Decorator to rate limit function calls"""
    
    def decorator(func):
        func._call_times = deque(maxlen=calls_per_minute)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Remove calls older than 1 minute
            while func._call_times and current_time - func._call_times[0] > 60:
                func._call_times.popleft()
            
            # Check rate limit
            if len(func._call_times) >= calls_per_minute:
                wait_time = 60 - (current_time - func._call_times[0])
                await asyncio.sleep(wait_time)
            
            # Record this call
            func._call_times.append(current_time)
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def cache_response(ttl: int = 3600):
    """Decorator to cache function responses"""
    
    def decorator(func):
        func._cache = {}
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = str(hash(str(args) + str(sorted(kwargs.items()))))
            
            # Check cache
            if cache_key in func._cache:
                cached_result, timestamp = func._cache[cache_key]
                if time.time() - timestamp < ttl:
                    return cached_result
                else:
                    del func._cache[cache_key]
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            func._cache[cache_key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator


# Usage example
if __name__ == "__main__":
    import asyncio
    
    async def demo_ai_models():
        """Demonstrate AI model management"""
        
        # Initialize manager
        manager = AIModelManager()
        
        # Generate text
        response = await manager.generate_text(
            "Explain the benefits of AI automation in business",
            model_name="gpt-4"
        )
        
        print("Generated Text:")
        print(response.content)
        print(f"Tokens used: {response.tokens_used}")
        print(f"Cost: ${response.cost:.4f}")
        print(f"Response time: {response.response_time:.2f}s")
        
        # Chat example
        messages = [
            {"role": "user", "content": "What are the key benefits of AI in customer service?"}
        ]
        
        chat_response = await manager.generate_chat_response(messages, "gpt-3.5-turbo")
        print(f"\nChat Response: {chat_response.content}")
        
        # Classification example
        classification = await manager.classify_text(
            "I'm very unhappy with your service!",
            ["positive", "negative", "neutral"]
        )
        print(f"\nSentiment: {classification.content}")
        
        # Usage statistics
        stats = manager.get_usage_stats()
        print(f"\nUsage Statistics: {stats}")
    
    # Run demo
    # asyncio.run(demo_ai_models())