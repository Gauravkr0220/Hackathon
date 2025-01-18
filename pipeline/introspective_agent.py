

from llama_index.core.tools import FunctionTool
from llama_index.agent.introspective import IntrospectiveAgentWorker
from llama_index.agent.introspective import (
    ToolInteractiveReflectionAgentWorker,
)
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgentWorker
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.llms import ChatMessage, MessageRole
from typing import Tuple
from llama_index.core.bridge.pydantic import Field
from llama_index.core.tools import FunctionTool
from utils.llm_api import llm_api

class IntrospectiveAgent:
    """Class for creating an introspective agent with tool-interactive reflection."""

    def __init__(self, perspective_tool: FunctionTool, verbose=True, with_main_worker=False):
        """
        Initialize the introspective agent.
        
        Parameters:
        - perspective_tool: The FunctionTool instance for evaluating text toxicity.
        - verbose: Whether to log detailed information during execution.
        - with_main_worker: Whether to include a main agent worker in the setup.
        """
        self.perspective_tool = perspective_tool
        self.verbose = verbose
        self.with_main_worker = with_main_worker

        
        self.agent = self._create_introspective_agent()

    def _create_introspective_agent(self):
        """Helper function to construct the introspective agent."""
       
        critique_agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=[self.perspective_tool],
            llm=OpenAI("gpt"),
            verbose=self.verbose,
        )

       
        correction_llm = OpenAI("gpt")

        
        def stopping_callable(critique_str: str) -> bool:
            return "[PASS]" in critique_str

        
        reflection_agent_worker = ToolInteractiveReflectionAgentWorker.from_defaults(
            critique_agent_worker=critique_agent_worker,
            critique_template=(
                "Please provide a toxicity critique for the provided text. "
                "If the toxicity score is less than 3.0 then please "
                "write '[PASS]' otherwise write '[FAIL]'. "
                "Here is the text:\n {input_str}"
            ),
            stopping_callable=stopping_callable,
            correction_llm=correction_llm,
            verbose=self.verbose,
        )

       
        main_agent_worker = (
            OpenAIAgentWorker.from_tools(
                tools=[], llm=OpenAI("gpt"), verbose=self.verbose
            )
            if self.with_main_worker
            else None
        )

       
        introspective_agent_worker = IntrospectiveAgentWorker.from_defaults(
            reflective_agent_worker=reflection_agent_worker,
            main_agent_worker=main_agent_worker,
            verbose=self.verbose,
        )

        
        chat_history = [
            ChatMessage(
                content="You are an assistant that generates safer versions of potentially toxic, user-supplied text.",
                role=MessageRole.SYSTEM,
            )
        ]

        
        return introspective_agent_worker.as_agent(
            chat_history=chat_history, verbose=self.verbose
        )

    def get_agent(self):
        """Return the initialized introspective agent."""
        return self.agent



from googleapiclient import discovery
from typing import Dict, Optional
import json
import os


class Perspective:
    """Custom class to interact with Perspective API."""

    attributes = [
        "toxicity",
        "severe_toxicity",
        "identity_attack",
        "insult",
        "profanity",
        "threat",
        "sexually_explicit",
    ]

    def __init__(self, api_key: Optional[str] = None) -> None:
        if api_key is None:
            try:
                api_key = os.environ["PERSPECTIVE_API_KEY"]
            except KeyError:
                raise ValueError(
                    "Please provide an api key or set PERSPECTIVE_API_KEY env var."
                )

        self._client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def get_toxicity_scores(self, text: str) -> Dict[str, float]:
        """Function that makes API call to Perspective to get toxicity scores across various attributes."""

        analyze_request = {
            "comment": {"text": text},
            "requestedAttributes": {
                att.upper(): {} for att in self.attributes
            },
        }

        response = (
            self._client.comments().analyze(body=analyze_request).execute()
        )
        try:
            return {
                att: response["attributeScores"][att.upper()]["summaryScore"][
                    "value"
                ]
                for att in self.attributes
            }
        except Exception as e:
            raise ValueError("Unable to parse response") from e


perspective = Perspective()




def perspective_function_tool(
    text: str = Field(
        default_factory=str,
        description="The text to compute toxicity scores on.",
    )
) -> Tuple[str, float]:
    """Returns the toxicity score of the most problematic toxic attribute."""

    scores = perspective.get_toxicity_scores(text=text)
    max_key = max(scores, key=scores.get)
    return (max_key, scores[max_key] * 100)




pespective_tool = FunctionTool.from_defaults(
    perspective_function_tool,
)


if __name__=="__main__":
       
    perspective_tool = FunctionTool.from_defaults(perspective_function_tool)

    
    introspective_agent_instance = IntrospectiveAgent(
        perspective_tool=perspective_tool, verbose=True, with_main_worker=False
    )

   
    agent = introspective_agent_instance.get_agent()

   
    response = agent.chat("Evaluate the following text for toxicity: 'This is an example text.'")
    print(response)