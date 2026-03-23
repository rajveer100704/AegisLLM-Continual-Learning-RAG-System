import google.generativeai as genai
from typing import Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from configs.config import settings
from utils.logger import logger, observe_latency_async

class LLMClient:
    """
    Production-grade Gemini API client with retry logic and error handling.
    """
    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL_NAME
        
        # Configure Gemini SDK
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": settings.TEMPERATURE,
                "max_output_tokens": settings.MAX_OUTPUT_TOKENS,
                "response_mime_type": "application/json", # Force JSON output
            }
        )
        logger.info(f"Initialized Gemini model: {self.model_name}")

    @observe_latency_async
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(f"Retrying LLM call... Attempt {retry_state.attempt_number}")
    )
    async def generate_async(self, prompt: str) -> str:
        """Generate response asynchronously with retry logic."""
        try:
            response = await self.model.generate_content_async(prompt)
            if not response.text:
                logger.error("LLM returned empty response")
                raise ValueError("Empty response from LLM")
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise
