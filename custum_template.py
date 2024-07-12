from langchain.prompts import PromptTemplate


def retriever_template():
    template = """You are a helpful assistant to resolve questions.
Given the context, provide an answer to the question. If you do not know the answer, respond with "I don't know the answer."
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question"
        ]
    )
    return prompt


def retriever_template2():
    template = """Given the context, provide an answer to the question.
    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question"
        ]
    )
    return prompt


def retriever_template3():
    template = """Given the context, provide an answer to the question. If you do not know the answer, respond with "I don't know the answer."
    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question"
        ]
    )
    return prompt


def retriever_only_context_template():
    template = """Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, provide an answer to the question.
    Question: {question}
    Answer: 
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question"
        ]
    )
    return prompt


def retriever_only_context_template2():
    template = """Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context provide an answer to the question.
    Question: {question}
    Answer: 
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question"
        ]
    )
    return prompt


def retriever_only_context_template3():
    template = """Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, provide an answer to the question.
    If you do not know the answer, respond with "I don't know the answer."
    Question: {question}
    Answer: 
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question"
        ]
    )
    return prompt


def knowledge_full_context_template():
    template = """Context information is below, but feel free to use all available knowledge to answer the question.
    ---------------------
    {context}
    ---------------------
    Given the context information and any additional knowledge, provide an answer to the question.
    Question: {question}
    Answer: 
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question"
        ]
    )
    return prompt


def knowledge_brief_answer_template():
    template = """Context information is below, but the response should be brief and directly answer the question without additional unnecessary information.
    ---------------------
    {context}
    ---------------------
    Question: {question}
    Answer: 
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question"
        ]
    )
    return prompt


def python_code_generation_template():
    template = """Context information is below, but feel free to use all available knowledge to write the Python code.
    ---------------------
    {context}
    ---------------------
    Given the context information and any additional knowledge, provide a Python code snippet to accomplish the following task.
    Task: {question}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question"
        ]
    )
    return prompt


def base_template():
    template = """You are a helpful assistant to resolve questions. 
    If you do not know the answer, respond with "I don't know the answer."
    Question:
    {question}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "question",
        ]
    )

    return prompt


def compare_template():
    template = """You are a helpful assistant tasked with evaluating the similarity between two given phrases. 
Please provide a score ranging from 0 to 1, where 0 indicates that the two phrases are not similar at all, and 1 indicates that they are identical.
Phrase 1:
{phrase1}
Phrase 2:
{phrase2}
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "phrase1",
            "phrase2"
        ]
    )
    return prompt


def compare_template2():
    template = """
You are an expert in analyzing and comparing the content of two phrases for their similarity in meaning, context, and factual accuracy. Your task is to provide a detailed comparison that focuses not only on the literal wording but also on the underlying ideas and accuracy of the information presented. Based on your analysis, assign a similarity score from 0 to 1, where 0 means no similarity at all and 1 means completely identical in meaning and factual content.

Consider the following phrases:

Phrase 1: "{phrase1}"
Phrase 2: "{phrase2}"

Given the context and the content of these phrases, please provide your similarity score along with a brief explanation of your evaluation, highlighting key similarities and differences in their meanings and factual accuracies.
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "phrase1",
            "phrase2"
        ]
    )
    return prompt


def answer_relevance_template():
    template = """
Given the following answer, please generate potential questions that this answer could be responding to. Focus on creating questions that directly relate to the core content of the answer. Avoid questions for which the answer would be incomplete or contain redundant information.

Answer: {answer}

Format your response as a list of questions: [Question1, Question2, ..., QuestionN]

Based on the given answer, please generate {n} questions and format them as a list.
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "answer",
            "n"
        ]
    )
    return prompt


def best_compare_template():
    template = """
    You are an expert in analyzing and comparing the content of two phrases for their similarity in meaning, context, and factual accuracy. Your task is to provide a detailed comparison that focuses not only on the literal wording but also on the underlying ideas and accuracy of the information presented. Based on your analysis, assign a similarity score from 0 to 1, where 0 means no similarity at all and 1 means completely identical in meaning and factual content.

    Consider the following phrases:

    Phrase 1: "{phrase1}"
    Phrase 2: "{phrase2}"

    Given the these phrases, please only return the similarity score.

    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "phrase1",
            "phrase2"
        ]
    )
    return prompt


def best_compare_template2():
    template = """
    You are an expert in analyzing and comparing the content of two phrases for their similarity in meaning, context, and factual accuracy. Your task is to provide a detailed comparison that focuses on the underlying ideas and accuracy of the information presented. Based on your analysis, assign a similarity score from 0 to 1, where 0 means no similarity at all and 1 means completely identical in meaning and factual content.

    Consider the following phrases:
    
    context: "{context}"
    Phrase 1: "{phrase1}"
    Phrase 2: "{phrase2}"

    Given the these phrases, please only return the similarity score.

    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "phrase1",
            "phrase2"
        ]
    )
    return prompt
