import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# Initialize OpenAI client with error handling
try:
    client = OpenAI()
    # Test the client
    client.models.list()
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise 