import os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import pymupdf
from llm_api import llm_api

class content_eval_agent:
    def __init__(self, model, api_key):
        self.llm=(lambda prompt: llm_api(prompt, model="", api_key=api_key))
        self.system_prompt= '''You are an expert in evaluating the relevance of research papers in conferences.
        In CVPR,conference mostly Computer vision and pattern recognition research related papers are submitted.
        In neurIPS,conference mostly machine learning, neuroscience, statistics, optimization, computer vision, natural language processing, life sciences, natural sciences, social science papers
        In EMNLP conference the research papers are on the topics related to natural language processing, machine language.
        In KDD conference research papers are on topics related to big data, data mining, machine learning.
        Key Distinction:
        CVPR: Focuses more on practical and application-driven works within computer vision.
        NeurIPS: Places greater emphasis on theoretical contributions and generalizable especially DEEP LEARNING methodologies, neural netowrks.
        EMNLP: Focuses more on NLP related tasks
        KDD: Focuses more on Big data, data-mining tasks.
        TMLR: Focuses on Algorithmic Recourse and Counterfactual Explanations,Applications of Large Language Models in Critical Domains,Graph Representation Learning,Privacy-Preserving Machine Learning

        The topics of CVPR includes the following but not limited to:--
        3D from multi-view and sensors
        3D from single images
        Adversarial attack and defense
        Autonomous driving
        Computational imaging
        Computer vision for social good
        Computer vision theory
        Datasets and evaluation
        Deep learning architectures and techniques
        Document analysis and understanding
        Efficient and scalable vision
        Embodied vision: Active agents, simulation
        Explainable computer vision
        Humans: Face, body, pose, gesture,
        movement
        Image and video synthesis and generation
        Low-level vision
        Machine learning (other than deep learning)
        Medical and biological vision, cell microscopy
        Multimodal learning
        Optimization methods (other than deeplearning)
        Photogrammetry and remote sensing
        Physics-based vision and shape-from-X
        Recognition: Categorization, detection,retrieval
        Representation learning
        Robotics
        Scene analysis and understanding
        Segmentation, grouping and shape analysis
        Self-& semi-& meta-& unsupervised learning
        Transfer/ low-shot/ continual/ long-tail learning
        Transparency, fairness, accountability, privacy and ethics in vision
        Video: Action and event understanding
        Video: Low-level analysis, motion, and
        tracking
        Vision + graphics
        Vision, language, and reasoning
        Vision applications and systems

        neurIPS topics:---
        Applications (e.g., vision, language, speech and audio, Creative AI)
        Deep learning (e.g., architectures, generative models, optimization for deep networks, foundation models, LLMs)
        Evaluation (e.g., methodology, meta studies, replicability and validity, human-in-the-loop)
        General machine learning (supervised, unsupervised, online, active, etc.)
        Infrastructure (e.g., libraries, improved implementation and scalability, distributed solutions)
        Optimization (e.g., convex and non-convex, stochastic, robust)
        Probabilistic methods (e.g., variational inference, causal inference, Gaussian processes)
        Reinforcement learning (e.g., decision and control, planning, hierarchical RL, robotics)
        Social and economic aspects of machine learning
        Theory (e.g.,learning theory, algorithmic game theory)
        Machine learning for sciences (e.g. climate, health, life sciences, physics, social sciences)

        EMNLP topics:---
        Language Modeling and Generation
        Machine Translation
        Information Extraction and Retrieval
        Sentiment Analysis and Opinion Mining
        Question Answering and Dialogue Systems
        Text Summarization
        Multimodal NLP
        Ethics and Fairness in NLP
        Low-Resource and Multilingual NLP
        Domain Adaptation and Transfer Learning in NLP

        KDD topics:---
        Data Mining and Knowledge Discovery:--
          Novel algorithms and techniques for data mining
          Mining patterns, rules, and logs
          Scalability and efficiency of data mining methods

        Machine Learning:---
          Development of new algorithms (supervised, unsupervised, semi-supervised, and reinforcement learning)
          Deep learning architectures and their applications
          Federated learning and optimization for distributed systems
          Big Data and Large-Scale Systems

        Systems for large-scale data analysis:--
            Parallel and distributed computing (cloud, MapReduce, Hadoop, Spark)
            Algorithmic and statistical techniques for large datasets
            Sampling, summarization, transformation, and integration techniques

        Data Science and Applications:----
          Methods for analyzing complex datasets, such as:
            Social networks
            Time series and sequences
            Streams and spatio-temporal data
            Text, web, and graph data
            IoT and multimedia data

        Computational advertising and recommender systems
        Bioinformatics, healthcare, and biological data analysis
        Business and finance applications

        TMLR topics:----
        theory, meta learning, gradient descent, learning rates, algorithmic complexity,
        stochastic optimization, theoretical analysis, optimization theory, regret minimization,
        convergence analysis,generalization, reproducibility, graph theory, online learning
        learning theory, hypothesis testing, statistical learning,
        information theory, gradient-based optimization

        

        Given a research paper,tell whether the paper domain is matching to the conference features and
        Give more attention to the domain and applications of abstract and introduction part, NOT the quality of the writing.
        Give a score out of 10 each for the relevance of the paper to each of the above conference topics separately.

        '''

        self.eval_prompt = ChatPromptTemplate.from_messages(
            [

                    ("system", self.system_prompt),


                    ( "user", "This is the content of the research paper:\n\n {content}.\n\n Please evaluate it and rate the topic allignment of paper out of 10 for each conference.",)
            ,
            ]
        )
        self.content_agent=self.eval_prompt | self.llm | StrOutputParser()

    def evaluate_content(self, content: str) -> str:
        evaluation = self.content_agent.invoke({"content": content})
        return evaluation



from math import ceil

def split_text(text, max_tokens=6000):
    """
    Splits text into smaller chunks to meet token limit.

    Args:
        text (str): The text to split.
        max_tokens (int): Maximum tokens allowed in each chunk.

    Returns:
        list: List of text chunks.
    """
    words = text.split()
    chunk_size = max_tokens
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def extract_and_chunk_paper(pdf_path, max_tokens=2000):
    """
    Extracts headings and content from a PDF and ensures they fit within token limits.

    Args:
        pdf_path (str): Path to the PDF file.
        max_tokens (int): Maximum tokens per chunk.

    Returns:
        list: List of dictionaries with headings and corresponding content chunks.
    """
    doc = pymupdf.open(pdf_path)
    headings_content = {}
    current_heading = "General"
    content_buffer = ""

    for page in doc:
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    text = " ".join(span["text"] for span in line["spans"]).strip()
                    font_size = line["spans"][0]["size"]

                    if font_size > 12:
                        if current_heading and content_buffer:
                            headings_content[current_heading] = content_buffer.strip()
                        current_heading = text
                        content_buffer = ""
                    else:
                        content_buffer += " " + text

    if current_heading and content_buffer:
        headings_content[current_heading] = content_buffer.strip()

    doc.close()


    chunked_output = []
    for heading, content in headings_content.items():
        chunks = split_text(content, max_tokens=max_tokens)
        for idx, chunk in enumerate(chunks):
            chunked_output.append({
                "heading": f"{heading} (Part {idx + 1})" if len(chunks) > 1 else heading,
                "content": chunk
            })
            break
        break

    return chunked_output



#2ndary agent*******************
class content_eval_agent2:
    def __init__(self, model, api_key):
        self.llm=(lambda prompt: llm_api(prompt, model="", api_key=api_key))
        content = {}
        content["Neurips"] = """"""
        content["EMNLP"] = """"""
        content["CVPR"] = """In CVPR,conference mostly Computer vision and pattern recognition research related papers are submitted.
        CVPR: Focuses more on practical and application-driven works within computer vision.
        The topics of CVPR includes the following but not limited to:--
                3D from multi-view and sensors
                3D from single images
                Adversarial attack and defense
                Autonomous driving
                Computational imaging
                Computer vision for social good
                Computer vision theory
                Datasets and evaluation
                Deep learning architectures and techniques
                Document analysis and understanding
                Efficient and scalable vision
                Embodied vision: Active agents, simulation
                Explainable computer vision
                Humans: Face, body, pose, gesture,
                movement
                Image and video synthesis and generation
                Low-level vision
                Machine learning (other than deep learning)
                Medical and biological vision, cell microscopy
                Multimodal learning
                Optimization methods (other than deeplearning)
                Photogrammetry and remote sensing
                Physics-based vision and shape-from-X
                Recognition: Categorization, detection,retrieval
                Representation learning
                Robotics
                Scene analysis and understanding
                Segmentation, grouping and shape analysis
                Self-& semi-& meta-& unsupervised learning
                Transfer/ low-shot/ continual/ long-tail learning
                Transparency, fairness, accountability, privacy and ethics in vision
                Video: Action and event understanding
                Video: Low-level analysis, motion, and
                tracking
                Vision + graphics
                Vision, language, and reasoning
                Vision applications and systems
        """
        content["KDD"] = """"""
        content["TMLR"] = """"""
        self.system_prompt= '''You are an intelligent paper evaluator designed to determine the most suitable journals or conferences for a research paper. Follow these steps to evaluate and recommend the best option:
        Step 1: Identify Top Journals/Conferences
        - Input: You will receive evaluation scores (out of 10) for five journals/conferences: NeurIPS, KDD, EMNLP, and CVPR, TMLR.
        - Task: Identify the two journals/conferences with the highest scores.
          - In case of a tie, prioritize the journals/conferences most aligned with the provided topics or themes.
        - Output: List the top two journals/conferences and their scores.

        Step 2: Evaluate Based on Topics
        - Input: A description of the research paper's topics and the themes typically discussed at the selected conferences/journals.
        - Task:
          - Analyze how well the research paper aligns with the focus areas of the top two journals/conferences.
          - Assign a score out of 10 to each journal/conference based on the relevance of the paper to their themes.
        - OUTPUT:
          - Provide the updated scores for the top two journals/conferences.
          - Highlight the topics that influenced the score adjustments.

        Step 3: Final Recommendation
        - Task:
          - Compare the updated scores and recommend the most suitable journal/conference for the paper.
          - Provide a detailed explanation justifying your choice within 100 words, including:
            - The alignment of the paper’s topics with the journal/conference themes.
            - The significance of the paper's contribution to the chosen venue's community.
        - Output:
          - The final recommendation of the journal/conference for submission.
          - Provide a detailed explanation justifying your choice within 100 words.

          The topics that are discussed in these conferences are :
                In CVPR,conference mostly Computer vision and pattern recognition research related papers are submitted.
                In neurIPS,conference mostly machine learning, neuroscience, statistics, optimization, computer vision, natural language processing, life sciences, natural sciences, social science papers
                In EMNLP conference the research papers are on the topics related to natural language processing, machine language.
                In KDD conference research papers are on topics related to big data, data mining, machine learning.
                Key Distinction:
                CVPR: Focuses more on practical and application-driven works within computer vision.
                NeurIPS: Places greater emphasis on theoretical contributions and generalizable especially DEEP LEARNING methodologies, neural netowrks.
                EMNLP: Focuses more on NLP related tasks
                KDD: Focuses more on Big data, data-mining tasks.

                The topics of CVPR includes the following but not limited to:--
                3D from multi-view and sensors
                3D from single images
                Adversarial attack and defense
                Autonomous driving
                Computational imaging
                Computer vision for social good
                Computer vision theory
                Datasets and evaluation
                Deep learning architectures and techniques
                Document analysis and understanding
                Efficient and scalable vision
                Embodied vision: Active agents, simulation
                Explainable computer vision
                Humans: Face, body, pose, gesture,
                movement
                Image and video synthesis and generation
                Low-level vision
                Machine learning (other than deep learning)
                Medical and biological vision, cell microscopy
                Multimodal learning
                Optimization methods (other than deeplearning)
                Photogrammetry and remote sensing
                Physics-based vision and shape-from-X
                Recognition: Categorization, detection,retrieval
                Representation learning
                Robotics
                Scene analysis and understanding
                Segmentation, grouping and shape analysis
                Self-& semi-& meta-& unsupervised learning
                Transfer/ low-shot/ continual/ long-tail learning
                Transparency, fairness, accountability, privacy and ethics in vision
                Video: Action and event understanding
                Video: Low-level analysis, motion, and
                tracking
                Vision + graphics
                Vision, language, and reasoning
                Vision applications and systems

                neurIPS topics:---
                Applications (e.g., vision, language, speech and audio, Creative AI)
                Deep learning (e.g., architectures, generative models, optimization for deep networks, foundation models, LLMs)
                Evaluation (e.g., methodology, meta studies, replicability and validity, human-in-the-loop)
                General machine learning (supervised, unsupervised, online, active, etc.)
                Infrastructure (e.g., libraries, improved implementation and scalability, distributed solutions)
                Optimization (e.g., convex and non-convex, stochastic, robust)
                Probabilistic methods (e.g., variational inference, causal inference, Gaussian processes)
                Reinforcement learning (e.g., decision and control, planning, hierarchical RL, robotics)
                Social and economic aspects of machine learning
                Theory (e.g.,learning theory, algorithmic game theory)
                Machine learning for sciences (e.g. climate, health, life sciences, physics, social sciences)

                EMNLP topics:---
                Language Modeling and Generation
                Machine Translation
                Information Extraction and Retrieval
                Sentiment Analysis and Opinion Mining
                Question Answering and Dialogue Systems
                Text Summarization
                Multimodal NLP
                Ethics and Fairness in NLP
                Low-Resource and Multilingual NLP
                Domain Adaptation and Transfer Learning in NLP

                KDD topics:---
                Data Mining and Knowledge Discovery:--
                  Novel algorithms and techniques for data mining
                  Mining patterns, rules, and logs
                  Scalability and efficiency of data mining methods

                Machine Learning:---
                  Development of new algorithms (supervised, unsupervised, semi-supervised, and reinforcement learning)
                  Deep learning architectures and their applications
                  Federated learning and optimization for distributed systems
                  Big Data and Large-Scale Systems

                Systems for large-scale data analysis:--
                    Parallel and distributed computing (cloud, MapReduce, Hadoop, Spark)
                    Algorithmic and statistical techniques for large datasets
                    Sampling, summarization, transformation, and integration techniques

                Data Science and Applications:----
                  Methods for analyzing complex datasets, such as:
                    Social networks
                    Time series and sequences
                    Streams and spatio-temporal data
                    Text, web, and graph data
                    IoT and multimedia data

                Computational advertising and recommender systems
                Bioinformatics, healthcare, and biological data analysis
                Business and finance applications

                TMLR topics:----
                theory, meta learning, gradient descent, learning rates, algorithmic complexity,
                stochastic optimization, theoretical analysis, optimization theory, regret minimization,
                convergence analysis,generalization, reproducibility, graph theory, online learning
                learning theory, hypothesis testing, statistical learning,
                information theory, gradient-based optimization

        Example Interaction
        Input:
        - Scores: NeurIPS: 8, KDD: 9, EMNLP: 6, CVPR: 7, TMLR: 5
        - Paper Topics: "Graph neural networks for spatio-temporal data analysis with applications in social networks and healthcare."

        Step 1 Output:
        Top journals/conferences:
        1. KDD (9)
        2. NeurIPS (8)

        Step 2 Output:
        - KDD: 9.5 (Strong focus on social networks, spatio-temporal data, and healthcare applications)
        - NeurIPS: 8.5 (Relevant for graph neural networks but less application-specific focus)

        Step 3 Output:
        RECOMMENDED JOURNAL: KDD
        Reason: The paper’s emphasis on spatio-temporal data and applications in social networks and healthcare aligns strongly with KDD’s focus on data mining and domain-specific applications.
                '''

        self.eval_prompt = ChatPromptTemplate.from_messages(
            [

                    ("system", self.system_prompt),


                    ( "user","This is the evaluation of research paper on the allignment of topic with each conference :\n\n {evaluation}.\n\n Please extract the top conference WITH HIGHEST SCORE and rate it out of 10. Give the whole explanation of choosing the best conference within 100 words.",)
            ,
            ]
        )
        self.content_agent=self.eval_prompt | self.llm | StrOutputParser()

    def evaluate_content(self, evaluation: str) -> str:
        evaluation = self.content_agent.invoke({"evaluation": evaluation})
        return evaluation
    



import glob
import re
#usable functions
def use_model(pdf_path):


    # Set the path to the folder containing the PDF files
    #pdf_path = '/content/drive/MyDrive/kgp/kdsh_data/Papers'

    # Get the list of all PDF files in the folder
    pdf_files = [pdf_path] # Adjust the pattern if needed


    model="llama3-70b-8192"
    api_key="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"

    #loop over all the papers present in the pdf_path directory
    output_prompts1=[]
    output_prompts2=[]
    file_identifiers=[]
    conference=[]
    ################################################################################################
    # Access files one by one
    i=1
    for pdf_file in pdf_files:
        #if(i>3):
        # break

        print("processing file: ",i)
        print(f"Processing file: {os.path.basename(pdf_file)}")
        # Add your processing code here, e.g., extracting text or other actions
        #1st agent
        headings_content = extract_and_chunk_paper(pdf_file)
        #call the agent using llm_api
        #content_evaluator = content_eval_agent(model, api_key, headings_content)

        agent_1=content_eval_agent(model=model, api_key=api_key)
        evaluation = agent_1.evaluate_content(headings_content)
        output_prompts1.append(' '.join(evaluation.strip().split(". ")[-2:]))

        #2nd agent
        agent_2=content_eval_agent2(model=model, api_key=api_key)
        evaluation_new = agent_2.evaluate_content(evaluation)

        output_prompts2.append(evaluation_new)



        file_name = os.path.basename(pdf_file)  # e.g., 'P062.pdf'
        file_identifier = os.path.splitext(file_name)[0]  # e.g., 'P062'
        file_identifiers.append(file_identifier)  # Add to the list

        i+=1

    
    
    df=pd.DataFrame({"Paper":[],"Conference":[],"Explanation_1":[],"Explanation_2":[]})
    df["Paper"]=file_identifiers
    #df["Conference"]=conference[:3]
    df["Explanation_1"]=output_prompts1
    df["Explanation_2"]=output_prompts2
    df.to_csv("output.csv")

if __name__=="__main__":
    use_model(r"/home/agniva-saha/Documents/KDSH/KDSH_2025/misc/output_0.pdf")
    