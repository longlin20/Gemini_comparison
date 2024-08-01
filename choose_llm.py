import vertexai
from langchain_google_vertexai import VertexAI


def initialize_vertexai(project_id, location: str) -> None:
    """
    Initialize the Vertex AI client with the given project ID and location.

    Args:
        project_id: The ID of the project.
        location (str): The location of the project.

    Returns:
        None
    """
    vertexai.init(project=project_id, location=location)


def create_gemini_llm(project_id, location: str, model_name: str, temperature: float) -> VertexAI:
    """
    Create a Gemini LLM (Language Model) instance.

    Args:
        project_id: The ID of the project.
        location (str): The location of the project.
        model_name (str): The name of the model.
        temperature (float): The temperature value.

    Returns:
        VertexAI: The created Gemini LLM instance.
    """
    initialize_vertexai(project_id, location)
    return VertexAI(model_name=model_name, temperature=temperature)


def gemini_llm(project_id, location: str) -> VertexAI:
    """
    Create a Gemini LLM instance with default parameters.

    Args:
        project_id: The ID of the project.
        location (str): The location of the project.

    Returns:
        VertexAI: The created Gemini LLM instance.
    """
    return create_gemini_llm(project_id, location, "gemini-1.0-pro-001", 0.2)


def gemini15_llm(project_id, location: str) -> VertexAI:
    """
    Create a Gemini LLM instance with the specified model name and temperature.

    Args:
        project_id: The ID of the project.
        location (str): The location of the project.

    Returns:
        VertexAI: The created Gemini LLM instance.
    """
    return create_gemini_llm(project_id, location, "gemini-1.5-pro-001", 0.2)