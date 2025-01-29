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
        Based on the past history of the conversation as below: \n{chat_history}\n
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
        self.facial_tagging_function = self.define_facial_tagging()
        print("DEBUG")
        print(self.facial_tagging_function)
        self.llm = self.llm.bind(functions=[{'facial_tagging': self.facial_tagging_function}])
        self.chain = self.define_chain()
        # if self.return_llm:
        #     self.return_llm = self.chain

    def define_facial_tagging(self):

        # fields = {
        #     expression: (float, Field(
        #         description=(f"'{expression}' facial tag based on user's query and "
        #                      "your response, ranging from a minimum of 0.0 "
        #                      "to a maximum of 0.99 (inclusive).")
        #     )) for expression in self.facial_expressions
        # }
        # tagging_expression = create_model("ExpressionsTagging", **fields)  # creating a base model

        class FacialExpressionTag(BaseModel):
            """Tag the piece of text with these facial expressions from 0.0 to 0.99"""
            neutral: float = Field(description="Neutral facial expression based on user's query and your response.")
            happy: float = Field(description="Happy facial expression based on user's query and your response.")
            sad: float = Field(description="Sad facial expression based on user's query and your response.")
            surprise: float = Field(description="Surprise facial expression based on user's query and your response.")
            anger: float = Field(description="Anger facial expression based on user's query and your response.")
            smile: float = Field(description="Smile facial expression based on user's query and your response.")

        return convert_pydantic_to_openai_function(FacialExpressionTag)                     # returning functions to bing

    def define_chain(self):
        # navel_chain = self.define_prompt() | self.llm | self.define_output_parser()
        navel_chain = self.define_prompt() | self.llm
        return navel_chain

    def __call__(self, query):

        response = self.chain.invoke({"input": query, "chat_history": self.chat_history})
        self.chat_history.append({"user": query, "assistant": response})
        if len(self.chat_history) > self.max_memory:
            self.chat_history.pop(0)  # FIFO memory buffer strategy
        return response


if __name__ == "__main__":

    # """
    # Usage #1:
    # Returns a conversational agent in the form of a callable object, to be used in order to have a smooth
    # multi-language conversation.
    # """
    #
    # print("=== USAGE EXAMPLE #1 ===")
    # conversational_agent = ConversableAgent(model_name="groq-mixtral")
    #
    # print("Starting conversation. Type 'exit' to end.")
    # while True:
    #     user_query = input("You: ")
    #     if user_query.lower() == "exit":
    #         print("Goodbye!")
    #         break
    #     print(conversational_agent(user_query))
    #
    #
    # # Usage 2
    # """
    # Usage #2:
    # Returns an llm model as a langchain runnable via GROQ provider. This can be used as an llm block to be implemented
    # with other means like function calling, tagging, extraction, and more complex chains.
    # """
    # print("=== USAGE EXAMPLE #2 ===")
    # llm_model = ConversableAgent(model_name="groq-mixtral",
    #                              return_model=True).return_llm
    #
    # # this is going to be a usable LLM returned as a LangChain Runnable, you can use it in more complex and dedicated ways
    #
    # response = llm_model.invoke("Whats the capital city of New Mexico?")
    # print(response.content)


    """
    Usage with Navel: 
    create a NavelConversableAgent to assess all features to be used to control a Navel Robot.
    """

    print("=== NAVEL USAGE EXAMPLE ===")
    navel_agent = NavelConversableAgent(model_name="deepseek")

    # using it as in a chat like - multiple queries in a raw
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        print(navel_agent(user_query))











