from utils.llm_api import llm_api
import re

def get_score(response : str) -> float:
    prompt = f"""You are given a text which evaluates a paper. The text mentions scores. You need to extract the final score from the text. The final score is a floating point number.
    You must not return anything else. Your response should be just the final score.(example: 7.5, 8.32, 9.0)
    Text:
    {response}
    Final score:"""
    response = llm_api(prompt, model="gpt", api_key="")
    # print(response)

    # extract the last number from the response
    score = re.findall(r"\d+\.\d+", response)
    # return last number
    return float(score[-1])

def get_final_conference(response : str) -> str:
    prompt = f"""You are given the results of expert analysis of which conference one should submit the paper to. You are given the review.
You must extract the conference name from the review. You should skip everything else and just return the conference name(examples: CVPR, NeurIPS, etc).
You must return exactly one conference name. Never return multiple conference names or None.
Review:-{response}
Final Conference:-"""
    response = llm_api(prompt, model="gpt", api_key="")
    return response

def get_review(response : str) -> str:
    prompt = f"""You are given the results of expert analysis of which conference one should submit the paper to. You are given the review.
You must extract the reasoning part of the review and the conference name from the review. You should skip the score parts and fully focus on the reasons given for the conference.
Review:-{response}
Extracted review:-"""
    response = llm_api(prompt, model="gpt", api_key="")
    return response