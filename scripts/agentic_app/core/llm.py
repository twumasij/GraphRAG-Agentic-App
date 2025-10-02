from typing import List, Optional
import requests
from pydantic import Field
from langchain.llms.base import LLM
from ..utils.monitoring import metrics
from ..utils.helpers import logger
from ..config.settings import config

class OllamaWrapper(LLM):
    """
    Simple wrapper for local Ollama server, ensuring we handle a single string prompt.
    """
    base_url: str = Field(default=config.OLLAMA_BASE_URL)
    model: str = Field(default=config.OLLAMA_MODEL)

    def __init__(self, base_url: str = None, model: str = None, **kwargs):
        super().__init__(**kwargs)
        if base_url:
            self.base_url = base_url
        if model:
            self.model = model

    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Called by LLM.invoke() or LLM.generate() with a single string prompt.
        """
        # The prompt must be a string (LangChain >0.1.7).
        if not isinstance(prompt, str):
            raise ValueError(
                f"OllamaWrapper _call expects a single string. Got {type(prompt)}."
            )

        # Check if this is a query that should generate a tool call
        is_tool_call_expected = False
        if "Human:" in prompt and "System:" in prompt:
            # This looks like a conversation with a human query
            lines = prompt.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("Human:") and i > 0 and i == len(lines) - 1:
                    # This is the last line and it's a human query
                    is_tool_call_expected = True
                    human_query = line.replace("Human:", "").strip().lower()
                    break

        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.OLLAMA_TEMPERATURE,
                "top_p": config.OLLAMA_TOP_P,
                "num_predict": config.OLLAMA_MAX_TOKENS
            }
        }
        if stop:
            data["stop"] = stop

        try:
            metrics.record_llm_call()
            logger.info(f"Ollama request → {self.base_url} / model={self.model}")
            logger.debug(f"Sending prompt to Ollama: {prompt}")
            
            # Add retry logic for robustness
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    resp = requests.post(f"{self.base_url}/api/generate", json=data, timeout=60)
                    resp.raise_for_status()
                    result = resp.json()
                    
                    if "response" not in result:
                        logger.error(f"Attempt {attempt+1}/{max_retries}: Unexpected Ollama response format: {result}")
                        if attempt == max_retries - 1:
                            raise ValueError(f"Unexpected Ollama response format: {result}")
                        continue
                    
                    response = result["response"]
                    
                    # Validate the response
                    if not response or not response.strip():
                        logger.warning(f"Attempt {attempt+1}/{max_retries}: Empty response from Ollama")
                        if attempt == max_retries - 1:
                            # If all attempts failed, return a fallback response
                            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your query."
                        continue
                    
                    # If we expect a tool call but don't see one, try to force it
                    if is_tool_call_expected and "TOOL:" not in response:
                        logger.warning("LLM did not return a tool call when one was expected")
                        
                        # Try to determine the appropriate tool based on the query
                        import re
                        if "shortest path" in human_query or "best route" in human_query:
                            # Extract airport codes
                            airports = re.findall(r'\b([A-Z]{3})\b', human_query.upper())
                            if len(airports) >= 2:
                                return f"TOOL: FindShortestPath {airports[0]},{airports[1]}"
                        elif "direct flight" in human_query or "direct flights" in human_query:
                            # Extract airport code
                            airports = re.findall(r'\b([A-Z]{3})\b', human_query.upper())
                            if airports:
                                return f"TOOL: FindDirectFlights {airports[0]}"
                        elif "importance" in human_query or "analyze" in human_query:
                            # Extract airport code
                            airports = re.findall(r'\b([A-Z]{3})\b', human_query.upper())
                            if airports:
                                return f"TOOL: AnalyzeAirportImportance {airports[0]}"
                    
                    logger.info(f"Ollama response received: {response[:100]}...")
                    return response
                    
                except (requests.exceptions.RequestException, ValueError) as e:
                    logger.error(f"Attempt {attempt+1}/{max_retries}: Request failed: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
            
            # This should not be reached due to the raises above, but just in case
            raise ValueError("Failed to get a valid response after multiple attempts")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {str(e)}")
            raise ValueError(f"Ollama request failed: {str(e)}")

def initialize_llm():
    print("Using Ollama for local LLaMA inference...")
    try:
        wrapper = OllamaWrapper(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL)
        r = requests.get(f"{wrapper.base_url}/api/version")
        r.raise_for_status()
        return wrapper
    except Exception as e:
        print(f"Warning: Could not initialize with model {config.OLLAMA_MODEL}: {str(e)}")
        print("Falling back to 'llama2' ...")
        return OllamaWrapper(model="llama2", base_url=config.OLLAMA_BASE_URL) 