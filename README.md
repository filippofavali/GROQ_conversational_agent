# ConversableAgent

The `ConversableAgent` is a Python-based library designed for interacting with language models (LLMs) in a conversational manner. It provides seamless integration with LLMs like GROQ Mixtral and OpenAI GPT-4 using the LangChain framework. The agent can act as a conversational assistant or provide access to the underlying LLM for advanced use cases.

---

## Features
- Multi-language conversational capabilities.
- Support for GROQ Mixtral and OpenAI GPT models.
- Flexible usage as a callable agent or a direct LLM provider.
- Environment variable management for API keys via `.env` file.

---

## Installation

### Clone the Repository
```bash
git clone <repository_url>
cd <repository_name>
```

### Set Up Virtual Environment

The project uses a `venv_droplet.yml` file for dependency management. Ensure you have `conda` installed, then create and activate the virtual environment:

```bash
conda env create -f venv_droplet.yml
conda activate <env_name>
```

---

## Configuration

### API Keys
Store the API keys in a `.env` file in the root directory of the project. For example:

```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

The program will automatically load these keys during initialization.

---

## Usage

### Example 1: Conversational Agent
The `ConversableAgent` can be used as a callable object for a multi-language conversational experience.

```python
from conversable_agent import ConversableAgent

# Create a conversational agent
conversational_agent = ConversableAgent(model_name="groq-mixtral")

print("Starting conversation. Type 'exit' to end.")
while True:
    user_query = input("You: ")
    if user_query.lower() == "exit":
        print("Goodbye!")
        break
    conversational_agent(user_query)
```

### Example 2: Direct LLM Access
You can also use `ConversableAgent` to directly access the underlying LLM as a LangChain `Runnable` for advanced workflows.

```python
from conversable_agent import ConversableAgent

# Get the LLM model directly
llm_model = ConversableAgent(model_name="groq-mixtral", return_model=True).return_llm

# Use the LLM model directly
response = llm_model.invoke("What's the capital city of New Mexico?")
print(response.content)
```

---

## Project Structure
- **`conversable_agent.py`**: Core implementation of the ConversableAgent class.
- **`.env`**: Stores API keys for LLM providers.
- **`venv_droplet.yml`**: Conda environment file for dependency management.

---

## Dependencies
The project requires the following dependencies:
- Python 3.8+
- `langchain`
- `dotenv`
- `pydantic`
- GROQ or OpenAI SDKs (depending on the selected LLM)

---

## Contributing
Feel free to fork the repository and submit pull requests for any improvements or bug fixes.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

