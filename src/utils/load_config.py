import os
from dotenv import load_dotenv
import yaml
from pyprojroot import here
from mistralai import Mistral
load_dotenv()


class LoadConfig:
    """
    A class for loading configuration settings, including OpenAI credentials.

    This class reads configuration parameters from a YAML file and sets them as attributes.
    It also includes a method to load OpenAI API credentials.

    Attributes:
        gpt_model (str): The GPT model to be used.
        temperature (float): The temperature parameter for generating responses.
        llm_system_role (str): The system role for the language model.
        llm_function_caller_system_role (str): The system role for the function caller of the language model.

    Methods:
        __init__(): Initializes the LoadConfig instance by loading configuration from a YAML file.
        load_openai_credentials(): Loads OpenAI configuration settings.
    """

    def __init__(self) -> None:
        with open(here("configs/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)
        self.gpt_model = app_config["gpt_model"]
        self.temperature = app_config["temperature"]
        self.llm_system_role = "You are a useful chatbot."
        self.llm_function_caller_system_role = app_config["llm_function_caller_system_role"]
        self.llm_system_role = app_config["llm_system_role"]

        self.load_mistral_credentials()

    def load_mistral_credentials(self):
        """
        Load and configure Mistral AI SDK settings.
        This function retrieves the Mistral API key from environment variables
        and initializes the Mistral client with it. This method should be called at application startup.

        Note:
            Ensure your .env file contains the MISTRAL_API_KEY.
            Example: MISTRAL_API_KEY="your-mistral-api-key"
        """
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")

        self.client = Mistral(api_key=api_key)
