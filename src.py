from pydantic import BaseModel, Field
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


if __name__ == "__main__":

    """
    Usage #1:
    Returns a conversational agent in the form of a callable object, to be used in order to have a smooth
    multi-language conversation.
    """

    print("=== USAGE EXAMPLE #1 ===")
    conversational_agent = ConversableAgent(model_name="groq-mixtral")

    print("Starting conversation. Type 'exit' to end.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        print(conversational_agent(user_query))

    # Usage 2
    """
    Usage #2:
    Returns an llm model as a langchain runnable via GROQ provider. This can be used as an llm block to be implemented 
    with other means like function calling, tagging, extraction, and more complex chains. 
    """
    print("=== USAGE EXAMPLE #2 ===")
    llm_model = ConversableAgent(model_name="groq-deepseek").return_llm()

    # Use the LLM model directly
    your_query = "Whats the capital city of New Mexico?"
    response = llm_model.invoke(your_query)
    print(response.content)







