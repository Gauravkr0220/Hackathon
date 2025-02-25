from COA_agent import CoAAgentWorker
import streamlit as st
from conference_editor_agent import get_conference
from aggregate import process_paper
from introspective_agent import IntrospectiveAgent
from utils.score import get_score, get_final_conference, get_review
from react_agent import React_Agent

def evaluate_paper(pdf_path):
    evaluation_1 = process_paper(pdf_path)
    evaluation_2 = get_conference(pdf_path)
    return evaluation_1, evaluation_2

if __name__ == "__main__":
    pdf_path = "misc/output_0.pdf"
    evaluation_1, evaluation_2 = evaluate_paper(pdf_path)
    print("Evaluation 1:", evaluation_1)
    print("Evaluation 2:", evaluation_2)