from typing import Dict
import inspect
import json
from inspect import Parameter
from pydantic import create_model
from .web_search import WebSearch
from typing import List, Dict
from .load_config import LoadConfig

cfg = LoadConfig()

class Apputils:

    @staticmethod
    def jsonschema(f) -> Dict:
        """
        Generate a JSON schema for the input parameters of the given function.
        Parameters:
        f (FunctionType): The function for which to generate the JSON schema.
        Returns:
        Dict: A dictionary containing the function name, description, and parameters schema.
        """
        kw = {n: (o.annotation, ... if o.default == Parameter.empty else o.default)
            for n, o in inspect.signature(f).parameters.items()}
        s = create_model(f'Input for `{f.__name__}`', **kw).model_json_schema()

        json_format = dict(
            type="function",
            function=dict(
                name=f.__name__,
                description=f.__doc__,
                parameters=s
            )
        )
        return json_format

    @staticmethod
    def wrap_functions() -> List:
        """
        Wrap several web search functions and generate JSON schemas for each.

        Returns:
            List: A list of dictionaries, each containing the function name, description, and parameters schema.
        """
        return [
            Apputils.jsonschema(WebSearch.retrieve_web_search_results),
            Apputils.jsonschema(WebSearch.web_search_text),
            Apputils.jsonschema(WebSearch.web_search_pdf),
            Apputils.jsonschema(WebSearch.web_search_image),
            Apputils.jsonschema(WebSearch.web_search_video),
            Apputils.jsonschema(WebSearch.web_search_news),
            Apputils.jsonschema(WebSearch.web_search_map),
        ]

    @staticmethod
    def execute_json_function(response) -> List:
        """
        Execute a function based on the response from an API call.

        Parameters:
            response: The response object from the API call.

        Returns:
            List: The result of the executed function.
        """
        # Extract function name and arguments from the response
        func_name: str = response.choices[0].message.tool_calls[0].function.name
        func_args: Dict = json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        )
        # Call the function with the given arguments
        if func_name == 'retrieve_web_search_results':
            result = WebSearch.retrieve_web_search_results(**func_args)
        elif func_name == 'web_search_text':
            result = WebSearch.web_search_text(**func_args)
        elif func_name == 'web_search_pdf':
            result = WebSearch.web_search_pdf(**func_args)
        elif func_name == 'web_search_image':
            result = WebSearch.web_search_image(**func_args)
        elif func_name == 'web_search_video':
            result = WebSearch.web_search_video(**func_args)
        elif func_name == 'web_search_news':
            result = WebSearch.web_search_news(**func_args)
        elif func_name == 'web_search_map':
            result = WebSearch.web_search_map(**func_args)
        else:
            raise ValueError(f"Function '{func_name}' not found.")
        return result

    @staticmethod
    def ask_llm_function_caller(gpt_model: str, temperature: float, messages: List, function_json_list: List):
        """
        Generate a response from an OpenAI ChatCompletion API call with specific function calls.

        Parameters:
            gpt_model (str): The name of the GPT model to use.
            temperature (float): The temperature parameter for the API call.
            messages (List): List of message objects for the conversation.
            function_json_list (List): List of function JSON schemas.

        Returns:
            The response object from the OpenAI ChatCompletion API call.
        """
        response = cfg.client.chat.complete(
            model = gpt_model,
            messages = messages,
            tools = function_json_list,
            tool_choice = "any",
            parallel_tool_calls = False,
            temperature=temperature
        )
        return response

    @staticmethod
    def ask_llm_chatbot(gpt_model: str, temperature: float, messages: List):
        """
        Generate a response from an OpenAI ChatCompletion API call without specific function calls.

        Parameters:
            gpt_model (str): The name of the GPT model to use.
            temperature (float): The temperature parameter for the API call.
            messages (List): List of message objects for the conversation.

        Returns:
            The response object from the OpenAI ChatCompletion API call.
        """
        response = cfg.client.chat.complete(
            model = gpt_model, 
            messages = messages,
            temperature=temperature
        )
        return response
    

if __name__ == "__main__":
    messages = [{
        "role": "user",
        "content": "who won the ipl 2025?"
    }]
    
    # Get the function tools and make the first LLM call
    helper_tools = Apputils.wrap_functions()
    first_llm_response = Apputils.ask_llm_function_caller(
        gpt_model=cfg.gpt_model,
        temperature=cfg.temperature,
        messages=messages,
        function_json_list=helper_tools
    )
    
    # Execute the function call
    function_result = Apputils.execute_json_function(first_llm_response)
    print("Function result:", function_result)
    print("\n\n\n")
    # Append the function result to the messages
    messages.append({
        "role": "assistant",
        "content": "",  # This can be empty as the tool call is what matters
        "tool_calls": [{
            "id": first_llm_response.choices[0].message.tool_calls[0].id,
            "type": "function",
            "function": {
                "name": first_llm_response.choices[0].message.tool_calls[0].function.name,
                "arguments": first_llm_response.choices[0].message.tool_calls[0].function.arguments
            }
        }]
    })
    
    messages.append({
        "role": "tool",
        "content": json.dumps(function_result),
        "tool_call_id": first_llm_response.choices[0].message.tool_calls[0].id
    })
    
    # Make the final LLM call with the function result
    final_response = Apputils.ask_llm_chatbot(
        gpt_model=cfg.gpt_model,
        temperature=cfg.temperature,
        messages=messages
    )
    
    print("Final response:", final_response.choices[0].message.content[0].text)
