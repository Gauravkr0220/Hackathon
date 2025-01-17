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