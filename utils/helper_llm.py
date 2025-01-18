from utils.llm_api import llm_api
from utils.splitter import chunker_function

def get_abstract(text: str) -> str:
    """
    Extracts the abstract from the text.
    """
    paragraphs = chunker_function(text)
    prompt = """You are given a text chunk from a research paper. You must return True if the chunk contains the abstract of the paper, and False otherwise.
    You must not return anything else. Your answer should be just a binary response(True/False)."""
    for paragraph, meta in paragraphs:
        new_text = f"{prompt}\n\nText: {paragraph} Answer:"
        response = llm_api(new_text)
        if "true" in response.lower() or "yes" in response.lower():
            return paragraph
    return ''

def get_tables(text: str) -> list[str]:
    """
    Extracts tables from the text.
    """
    tables = []
    paragraphs = chunker_function(text)
    prompt = """You are given a text chunk from a research paper. You must return True if the chunk contains a table, and False otherwise.
    You must not return anything else. Your answer should be just a binary response(True/False)."""
    for paragraph, meta in paragraphs:
        new_text = f"{prompt}\n\nText: {paragraph} Answer:"
        response = llm_api(new_text)
        if "true" in response.lower() or "yes" in response.lower():
            tables.append(paragraph)
    return tables

def filtering_agent(abstract: str, abstract_2: str) -> bool:
    """
    The abstract of two papers are given as input. The llm reasons about the two abstracts.
    Its follows a step by step chain of thought to reason about the two abstracts.
    Finally it tell whether the two papers are based on similar topics or not.
    This is because we will use the second paper to check hoe good the first paper is.
    """
    prompt = f"""You are given two abstracts from research papers. You must reason through the abstracts step by step to determine if the two papers are based on similar topics and ideas.
    You msut understand whether the second paper can be used to check the validity of the first paper. Finally at the end you must return True if the two papers are based on similar topics and ideas, and False otherwise.
    Abstract 1: {abstract}
    Abstract 2: {abstract_2}
    """
    response = llm_api(prompt)
    response = ' '.join(response.strip().split("\n")[-1].split(". ")[-2:])
    if "true" in response.lower() or "yes" in response.lower():
        return True
    return False