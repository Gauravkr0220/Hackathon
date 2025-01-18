from utils.llm_api import llm_api

def splitter_function(text: str) -> list[tuple[str, dict]]:
    """
    Simple splitter that divides text into sentences.
    
    Args:
        text (str): The input text to split.
    
    Returns:
        list[tuple[str, dict]]: A list of tuples containing each sentence and an empty metadata dictionary.
    """
    sentences = text.split('. ')
    return [(sentence.strip(), {}) for sentence in sentences if sentence]

def chunker_function(text: str) -> list[tuple[str, dict]]:
    """
    Paragraph chunker that divides text into paragraphs.
    """
    paragraphs = text.split('\n\n')
    return [(paragraph.strip(), {}) for paragraph in paragraphs if paragraph]

def get_abstract(text: str) -> str:
    """
    Extracts the abstract from the text.
    """
    paragraphs = chunker_function(text)
    for paragraph, meta in paragraphs:
        if 'abstract' in paragraph.lower() or len(paragraph) > 200:
            return paragraph
    return ''

def get_tables(text: str) -> list[str]:
    """
    Extracts tables from the text.
    """
    tables = []
    paragraphs = chunker_function(text)
    for paragraph, meta in paragraphs:
        if 'table' in paragraph.lower():
            tables.append(paragraph)
    return tables