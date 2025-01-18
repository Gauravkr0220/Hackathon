import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import pymupdf
from utils.llm_api import llm_api
class content_eval_agent:
    def __init__(self, model, api_key, content):
        self.llm=(lambda prompt: llm_api(prompt, model="gpt", api_key=api_key))
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

        Finally tell which conference the paper can be accepted to based on the scores.
        In the last line you must mention the best two conferences. For example, The paper should be submitted to CVPR or neurIPS, TMLR or KDD, etc.
        '''

        self.eval_prompt = ChatPromptTemplate.from_messages(
            [

                    ("system", self.system_prompt),


                    ( "user", "This is the content of the research paper:\n\n {content}.\n\n Please evaluate it and rate the overall quality of paper out of 10. In the last line you must mention the best two conferences. For example, The paper should be submitted to CVPR or neurIPS, TMLR or KDD, etc."
,)
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


from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import pymupdf
class content_eval_agent_2:
    def __init__(self, model, api_key, content):
        self.llm=(lambda prompt: llm_api(prompt, model="gpt", api_key=api_key))
        self.system_prompt= '''You are an intelligent paper evaluator designed to determine the most suitable journals or conferences for a research paper. Follow these steps to evaluate and recommend the best option:

Step 1: Identify Top Journals/Conferences
- Input: You will receive the top two conferences to which the paper shoudl be submitted.
Step 1: Analyse the research paper and and only the two conferences provided. Do not think about other conferences.

Step 2: Evaluate Based on Topics
- Input: A description of the research paper's topics and the themes typically discussed at the selected conferences/journals.
- Task:
  - Analyze how well the research paper aligns with the focus areas of the two journals/conferences.
  - Assign a score out of 10 to each journal/conference based on the relevance of the paper to their themes.
- Output:
  - Provide the updated scores for the top two journals/conferences.
  - Highlight the topics that influenced the score adjustments.

Step 3: Final Recommendation
- Task:
  - Compare the updated scores and recommend the most suitable journal/conference for the paper.
  - Provide a detailed explanation justifying your choice, including:
    - The alignment of the paper’s topics with the journal/conference themes.
    - The significance of the paper's contribution to the chosen venue's community.
- Output:
  - The final recommendation of the journal/conference for submission.
  - A concise reason supporting the recommendation.

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
- Scores: NeurIPS or KDD
- Paper Topics: "Graph neural networks for spatio-temporal data analysis with applications in social networks and healthcare."

Step 1 Output:
Top journals/conferences:
1. KDD
2. NeurIPS

Step 2 Output:
- KDD: 9.5 (Strong focus on social networks, spatio-temporal data, and healthcare applications)
- NeurIPS: 8.5 (Relevant for graph neural networks but less application-specific focus)

Step 3 Output:
Recommended Journal: KDD
Reason: The paper’s emphasis on spatio-temporal data and applications in social networks and healthcare aligns strongly with KDD’s focus on data mining and domain-specific applications.
        '''

        self.eval_prompt = ChatPromptTemplate.from_messages(
            [

                    ("system", self.system_prompt),


                    ( "user", "This is the evaluation of the research paper on journals CVPR, EMNLP, Neurips, KDD, TMLR :\n\n {evaluation}.\n\n You are given the top two choices and you must reason through them and analyse and rate them out of 10 for only the two conferences provided. Finally output the best conference to publish the paper to.",)
            ,
            ]
        )
        self.content_agent=self.eval_prompt | self.llm | StrOutputParser()

    def evaluate_content(self, evaluation: str) -> str:
        evaluation = self.content_agent.invoke({"evaluation": evaluation})
        return evaluation



from math import ceil

def split_text(text, max_tokens=7500):
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

def extract_and_chunk_paper(pdf_path, max_tokens=7500):
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


def get_conference(pdf_path):
   # pdf_path = "/content/drive/MyDrive/kgp/kdsh_data/Reference/Publishable/TMLR/R014.pdf"
    #headings_content = extract_and_chunk_paper(pdf_path)


    # for heading, content in headings_content.items():
    #     print(f"== {heading} ==")
    #     print(content if content else "No content found.")
    #     print("\n")
    headings_content = extract_and_chunk_paper(pdf_path)


    # for heading, content in headings_content.items():
    #     print(f"== {heading} ==")
    #     print(content if content else "No content found.")
    #     print("\n")
    output = ""

    model="llama3-70b-8192"
    api_key="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"
    content_evaluator = content_eval_agent(model, api_key, headings_content)
    evaluation = content_evaluator.evaluate_content(headings_content)
    print(evaluation)
    output = evaluation + "\n"
    evaluation = evaluation.strip().split(". ")[-1]
    print("FIRST : ", evaluation)
    content_evaluator = content_eval_agent_2(model, api_key, evaluation)
    evaluation_new = content_evaluator.evaluate_content(evaluation)
    print("SECOND : ", evaluation_new)

    output = evaluation_new
    return output



