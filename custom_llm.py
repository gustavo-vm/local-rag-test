import requests
from langchain.llms.base import BaseLLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import Generation, LLMResult
from langchain.schema.embeddings import Embeddings
from langchain.pydantic_v1 import BaseModel

class OllamaCustomEmbedding(BaseModel, Embeddings):

    api_url = "http://127.0.0.1:11434/api/embeddings"
    model: str = "mistral"

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        
        embbeding_list = []
        for i, text in enumerate(texts):
        
            data = {
                "model": self.model,
                "prompt": text,
            }
                    
            response = requests.post(self.api_url, json=data)
            if response.status_code == 200:
                embedding = response.json()["embedding"]
                embbeding_list.append(embedding)
            else:
                    # Handle HTTP errors (e.g., by raising an exception).
                    response.raise_for_status()

        return embbeding_list

    def embed_query(self, text: str) -> List[float]:

        return self.embed_documents(texts=[text])[0]


class OllamaCustomLLM(BaseLLM):

    api_url = "http://127.0.0.1:11434/api/generate"
    api_key = ''
    model =  'mistral'

    @property
    def _llm_type(self) -> str:
        return "custom"
        
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[: self.n]
    

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        
        print(prompts)
        
        text = self.generate(prompt=prompts[0])

        llm_output = {
            "token_usage": 1,
            "model_name": self.model,
        }

        return LLMResult(
            generations=[
                [Generation(text=text)]
            ],
            llm_output=llm_output,
        )
    
    

    def generate(self, prompt, max_tokens=100, **kwargs):
        # Prepare the request data according to your LLM's API specification.
        data = {
            "model": self.model,
            "prompt": prompt[0],
            "stream": False
        }
                
        response = requests.post(self.api_url, json=data)

        # Check for errors in the API response and handle them appropriately.
        if response.status_code == 200:
            completion = response.json()  # Adapt this line to match the actual response format.
            llm_output = {
                "token_usage": 1,
                "model_name": self.model,
            }

            return LLMResult(
                generations=[
                    [Generation(text=completion["response"])]
                ],
                llm_output=llm_output,
            )
        else:
            # Handle HTTP errors (e.g., by raising an exception).
            response.raise_for_status()

# # Example usage:
# api_url = "http://localhost:11434/api/generate"
# api_key = "your_llm_api_key"

# # Create an instance of your custom LLM wrapper.
# custom_llm = OllamaCustomLLM()#(api_url=api_url, api_key=api_key, model='mistral')

# # Use the `generate` method to get a completion from your LLM.
# print(custom_llm("Explain machine learning in one paragraph"))


# embeddings = OllamaCustomEmbedding()
# print(embeddings.embed_query("This is a test query."))

