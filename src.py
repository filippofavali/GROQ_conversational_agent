from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from pydantic import BaseModel, Field, create_model
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os


class ConversableAgent:

    def __init__(self, model_name:str="groq-mixtral",
                 temperature:float=0.0,
                 system_role:str="You are a helpful conversational counterpart",
                 memory_length:int=5,
                 ):

        self.model_name = model_name.lower()
        self.temperature = temperature
        self.system_role = system_role
        self.chat_history = []                                          # To store the conversation history
        self.max_memory = memory_length                                 # FIFO handling of memory buffer

        # Load environment variables from .env file
        load_dotenv()

        try:
            if "groq" in self.model_name:
                os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
            elif "openai" in self.model_name:
                os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
            else:
                raise NotImplemented
        except:
            TypeError("API_KEY not found in environment variables. Please set it in a .env file.")

        self.llm = self.load_llm()

    def return_llm(self):
        return self.llm

    def load_llm(self):
        if "mixtral" in self.model_name:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model="mixtral-8x7b-32768",
                temperature=self.temperature,
            )
        elif "llama" in self.model_name:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model="llama-3.3-70b-specdec",
                temperature=self.temperature,
            )
        elif "deepseek" in self.model_name:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model="deepseek-r1-distill-llama-70b",
                temperature=self.temperature,
            )
        elif self.model_name == "openai" in self.model_name:
            from langchain.chat_models import ChatOpenAI
            return ChatOpenAI(
                model="gpt-4",
                temperature=self.temperature
            )
        else:
            raise NotImplementedError("Model not supported")

    def define_prompt(self):
        user_input = """
        Based on the past history of the conversation as below (bypass if empty): \n{chat_history}\n
        Kindly answer to user query, in the language used by the user:\n {input}
        """

        return ChatPromptTemplate([
            ("system", self.system_role),
            ("user", user_input),
        ])

    def define_output_parser(self):
        return StrOutputParser()

    def __call__(self, query):
        prompt = self.define_prompt()
        parser = self.define_output_parser()

        chain = prompt | self.llm | parser

        response = chain.invoke({"input": query, "chat_history": self.chat_history})
        self.chat_history.append({"user": query, "assistant": response})
        if len(self.chat_history) > self.max_memory:
            self.chat_history.pop(0)                                # FIFO memory buffer strategy
        return response


class NavelConversableAgent(ConversableAgent):

    def __init__(self,
                 model_name:str="groq-mixtral",
                 temperature:float=0.0,
                 system_role:str=("You are Navel, an helpful robot assistant with a pair of arms, wheels to move around "
                                  " and a very sophisticated face to talk to the user and look around."),
                 memory_length: int = 5,
                 facial_expressions:list[str]=["neutral", "happy", "sad", "surprise", "anger", "smile"],
                ):

        super().__init__(model_name,
                         temperature,
                         system_role,
                         memory_length)

        self.facial_expressions = facial_expressions
        self.agent_memory = ["none"]
        self.user_memory = ["none"]
        self.chat_memories = [self.agent_memory, self.user_memory]
        self.main_chain = self.define_chain()                               # chain to answer the user
        self.memory_tagging_chain = self.define_emotional_tagging_chain()   # chain to tag agent and user emotions

    def __call__(self, query):

        response = self.main_chain.invoke({"input": query, "chat_history": self.user_memory})
        self.update_memory(query=query, response=response)

        return response

    def update_memory(self, query:str, response:str):

        self.agent_memory.append(query)
        self.user_memory.append(response)
        [memory.pop(0) for memory in self.chat_memories if len(memory) > self.max_memory]       # FIFO memory handling

    def define_facial_tagging(self):

        class FacialExpressionTag(BaseModel):
            """Tag the piece of text with these facial expressions from 0.0 to 0.99"""
            neutral: float = Field(description="Neutral facial expression based on user's query and your response.")
            happy: float = Field(description="Happy facial expression based on user's query and your response.")
            sad: float = Field(description="Sad facial expression based on user's query and your response.")
            surprise: float = Field(description="Surprise facial expression based on user's query and your response.")
            anger: float = Field(description="Anger facial expression based on user's query and your response.")
            smile: float = Field(description="Smile facial expression based on user's query and your response.")

        # return convert_pydantic_to_openai_function(FacialExpressionTag)               # returning functions to bind
        return FacialExpressionTag

    def define_chain(self):
        navel_chain = self.define_prompt() | self.llm | self.define_output_parser()
        return navel_chain

    def memory_prompt(self):
        user_input = """
        Based on the history contained into memory buffer: \n{memory_buffer}
        You are asked to control your own face via emotion tagging facial expression.
        """

        return ChatPromptTemplate([
            ("system", self.system_role),
            ("user", user_input),
        ])

    def define_emotional_tagging_chain(self):
        return self.memory_prompt() | self.llm.with_structured_output(self.define_facial_tagging())

    def return_agent_emotion(self):
        return self.memory_tagging_chain.invoke({"memory_buffer": self.agent_memory})

    def return_user_emotion(self):
        return self.memory_tagging_chain.invoke({"memory_buffer": self.user_memory})


if __name__ == "__main__":

    """
    Usage with Navel: 
    create a NavelConversableAgent to assess all features to be used to control a Navel Robot.
    """

    print("=== NAVEL USAGE EXAMPLE ===")
    navel_agent = NavelConversableAgent(model_name="llama")

    # using it as in a chat like - multiple queries in a raw, every n_step query the Agent returns an emotional status
    n_step = 2
    calls_counter = 0
    print("Simulation starting. Type exit to quit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        print(navel_agent(user_query))
        if calls_counter % n_step == 0:
            print("Navel emotions: ", navel_agent.return_agent_emotion())
            print("User emotions: ", navel_agent.return_user_emotion())
        calls_counter += 1










