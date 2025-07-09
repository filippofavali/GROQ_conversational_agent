from groq import Groq
from dotenv import load_dotenv
import pandas as pd


class PromptsTemplate:
    """
    A class to manage prompts templates for LLM clients.
    """

    _system_message = """You are a helpful assistant, leading a set of conversation with a user suffering of a cognitive impairment."""
    _user_message = """
    Your task is to answer to the user based in the context provided. 
    If the user response is not related to the context, respond with something generic as a personal assistant.
    This is the user response: {user_response}
    This is the context you have: {context}

    What is your response to the user? ANswer in the same language you receive the message.
    """

    @staticmethod
    def get_user_message()-> str:
        """
        Get the user message formatted for the LLM client.
        """
        return PromptsTemplate._user_message
    
    @staticmethod
    def get_system_message()-> str:
        """
        Get the system message formatted for the LLM client.
        """
        return PromptsTemplate._system_message


class GroqLLMSRegistry:

    """
    A class to manage a registry of LLM clients in a centraliz
    """

    _groq_registry = {
        "mixtral": "mixtral-8x7b-32768",
        "mistral-saba": "mistral-saba-24b",
        "llama3.1-instant": "llama-3.1-8b-instant",
        "llama3-70b": "llama3-70b-8192",
        "llama3.3": "llama-3.3-70b-versatile",
        "deepseek-llama": "deepseek-r1-distill-llama-70b",
        "deepseek-qwen": "deepseek-r1-distill-qwen-32b",
        "qwen-2": "qwen-2.5-32b",
        "image": "llama-3.2-90b-vision-preview"
    }

    _openai_registry = {
        "gpt-4": "gpt-4",
        "gpt-4-turbo": "gpt-4-turbo",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
    }

    @staticmethod
    def _get_groq_model_name(model_name: str) -> str:
        assert model_name in GroqLLMSRegistry._groq_registry, NotImplementedError(
            f"Model {model_name} is not supported by GroqLLMSRegistry. Supported models: {list(GroqLLMSRegistry._groq_registry.keys())}"
        )
        return GroqLLMSRegistry._groq_registry.get(model_name)
        
    @staticmethod
    def _get_openai_model_name(model_name: str) -> str:
        assert model_name in GroqLLMSRegistry._openai_registry, NotImplementedError(
            f"Model {model_name} is not supported by OpenAILLMSRegistry. Supported models: {list(GroqLLMSRegistry._openai_registry.keys())}"
        )
        return GroqLLMSRegistry._openai_registry.get(model_name)


class LLMClient:

    """
    A class to manage LLM clients for different providers.
    """

    def __init__(self, **model_parameters):

        load_dotenv()

        assert 'model_name' in model_parameters.keys(), "Model name must be provided in the model parameters."

        model_name = model_parameters.get("model_name")

        self.provider = model_name.split("/")[0] 
        self.model_name = self.get_model_name(model_name.split("/")[1] if "/" in model_name else model_name)

        self.temperature = model_parameters.get("temperature", 0.0)
        self.max_tokens = model_parameters.get("max_tokens", 4096)
        self.top_p = model_parameters.get("top_p", 1.0)
        self.stream = model_parameters.get("stream", False)
        self.stop = model_parameters.get("stop", None)

        self.client = Groq()

        self.usage_metrics = None     # empty and store this with whatever the client returns 

    def get_model_name(self, model_name) -> str:
        """        
        Get the model name based on the instanciation's parameters. 
        """
        if self.provider == "groq":
            return GroqLLMSRegistry._get_groq_model_name(model_name)
        elif self.provider == "openai":
            return GroqLLMSRegistry._get_openai_model_name(model_name)
        else:
            raise NotImplementedError(f"Provider {self.provider} is not supported.")
        
    def update_usage_metrics(self, **usage_metrics):
        """
        Update the usage metrics with the provided metrics.
        This method can be used to store usage metrics returned by the LLM client.
        """
        if self.usage_metrics is None:
            self.usage_metrics = pd.DataFrame([usage_metrics])
        else:
            new_row = pd.DataFrame([usage_metrics])
            self.usage_metrics = pd.concat([self.usage_metrics, new_row], ignore_index=True)

    def __call__(self, user_message: str, system_message: str = "You are a helpful assistant.", force_json: bool = False, **kwargs):
        """
        Call the LLM client to generate a response based on the user and system messages.
        """
        assert user_message, "User message cannot be empty when calling the LLM client."

        # Merge default parameters with overrides from kwargs
        parameters = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": kwargs.get("stream", self.stream),
            "stop": kwargs.get("stop", self.stop),
        }

        if self.provider == 'groq':
            # Parameters for Groq completion request
            request_groq_completion = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "temperature": parameters["temperature"],
                "max_completion_tokens": parameters["max_completion_tokens"],
                "top_p": parameters["top_p"],
                "stream": parameters["stream"],
                "stop": parameters["stop"],
            }

            if force_json:
                request_groq_completion["response_format"] = {"type": "json_object"}

            completion = self.client.chat.completions.create(**request_groq_completion)

            self.update_usage_metrics(**dict(completion.usage))
            return completion.choices[0].message.content

        elif self.provider == 'openai':
            raise NotImplementedError("OpenAI provider is not implemented yet.")
        else:
            raise NotImplementedError(f"Provider {self.provider} is not supported.")
    

if __name__ == "__main__":

    # Example usage
    llm_parameters = {
        "model_name": "groq/llama3.1-instant",
    }

    llm_client = LLMClient(**llm_parameters)
    user_message = PromptsTemplate.get_user_message()
    system_message = PromptsTemplate.get_system_message()

    context = "La foto ritrae due persone in montagna. Una donna ed un uomo."
    user_transcript = "Nella foto ci sono io da piccolo."

    response = llm_client(
        user_message=user_message.format(user_response=user_transcript, context=context),
        system_message=system_message,
        temperature=0.7,
        top_p=0.9
    )

    print("LLM Response:\n", response)

    print(llm_client.usage_metrics)

