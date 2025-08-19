import os
import re
import ast
import logging
from typing import Dict, Any
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from utils import (
    ocr_process_input,
    process_conversation_search,
    retrieve_qa,
    retrieve_docs_hybrid,
    retrieve_docs_manual,
    generate,
    log_blob,
    log_local,
)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = Flask(__name__)

load_dotenv('./keys.env')

def load_course_config(course: str) -> None:
    global prompts
    if 'ds100' in course:
        import prompts.ds100_multiturn_prompts as prompts
        course = 'ds100'
    elif 'ds8' in course:
        import prompts.ds8_multiturn_prompts as prompts
        course = 'ds8'
    elif 'cs61a' in course:
        import prompts.cs61a_multiturn_prompts as prompts
        course = 'cs61a'
    else:
        raise ValueError(f"Unsupported course: {course}")
    load_dotenv(f'configs/{course}.env', override=True)

def get_env_list(key: str) -> list:
    return ast.literal_eval(os.getenv(key, '[]'))

#Miloh Office Hours Extension
@app.route('/miloh', methods=['POST'])
def miloh():
    """
    Minimal additional endpoint to handle the new Office Hour extension's JSON:
    {
      "assignment": "string",
      "question": "string",
      "location": "string",
      "description": "string",
      "chat": ["string", ...]
    }
    Calls the LLM and returns the response.
    """
    if request.headers.get('Authorization') != os.getenv('API_KEY'):
        logger.warning('Unauthorized access attempt')
        return jsonify(error='Unauthorized'), 401

    # Get input data and load the course config    
    input_dict = request.json or {}
    logger.info('Received input: %s', input_dict)
    course = 'ds100_miloh'
    load_course_config('ds100_miloh')

    assignment_categories = get_env_list('ASSIGNMENT_CATEGORIES')
    content_categories = get_env_list('CONTENT_CATEGORIES')
    logistics_categories = get_env_list('LOGISTICS_CATEGORIES')
    worksheet_categories = get_env_list('WORKSHEET_CATEGORIES')

    # Construct Ed-like payload
    conversation_history = []

    # Put the main ticket info (assignment, question, description)
    # as the first "student" turn:
    initial_text = f"Assignment: {input_dict.get('assignment','')}\n" \
                   f"Question: {input_dict.get('question','')}\n" \
                   f"Description: {input_dict.get('description','')}"

    conversation_history.append({
        "user_role": "student",      # treat this entire first chunk as "student" text
        "text": initial_text,
        "document": None              # No images (so no OCR needed)
    })

    # Then put each chat message as additional "student" turns
    for c in input_dict.get('chat', []):
        conversation_history.append({
            "user_role": "student",
            "text": c,
            "document": None
        })

    # For thread_title, combine assignment + question:
    thread_title = f"{input_dict.get('assignment','')} — {input_dict.get('question','')}"
    processed_conversation = ocr_process_input(
        thread_title=thread_title,
        conversation_history=conversation_history,
    )
    logger.info("Processed conversation: %s", processed_conversation)

    processed_conversation_search = process_conversation_search(
        processed_conversation=processed_conversation,
        prompt_summarize=prompts.get_summarize_conversation_prompt(processed_conversation[:-1])
    )
    logger.info('Processed (summarized) conversation for search: %s', processed_conversation_search)

    # QA retrieval
    top_k = int(os.getenv('QA_TOP_K', '3'))
    retrieved_qa_pairs = retrieve_qa(conversation=processed_conversation_search, top_k=top_k)
    logger.info('Retrieved QA pairs: %s', retrieved_qa_pairs)

    question_category = 'Homeworks'

    # Hybrid document retrieval
    retrieved_docs_hybrid = 'none'
    if question_category in content_categories:
        retrieved_docs_hybrid = retrieve_docs_hybrid(
            text=processed_conversation_search,
            index_name=os.getenv('CONTENT_INDEX_NAME'),
            top_k=int(os.getenv('CONTENT_INDEX_TOP_K', '1')),
            semantic_reranking=True
        )
    elif question_category in logistics_categories:
        retrieved_docs_hybrid = retrieve_docs_hybrid(
            text=processed_conversation_search,
            index_name=os.getenv('LOGISTICS_INDEX_NAME'),
            top_k=int(os.getenv('LOGISTICS_INDEX_TOP_K', '1')),
            semantic_reranking=False
        )
    elif question_category in worksheet_categories:
        retrieved_docs_hybrid = retrieve_docs_hybrid(
            text=processed_conversation_search,
            index_name=os.getenv('WORKSHEET_INDEX_NAME'),
            top_k=int(os.getenv('WORKSHEET_INDEX_TOP_K', '1')),
            semantic_reranking=True
        )
    logger.info('Retrieved hybrid documents: %s', retrieved_docs_hybrid)

    # Manual document retrieval
    problem_list_manual = selected_doc_manual = retrieved_docs_manual = 'none'
    if question_category in (assignment_categories + worksheet_categories):
        question_info = re.sub(
            r"\n+",
            " ",
            f"{question_category} "
            f"{input_dict.get('assignment', '')} "
            f"{input_dict.get('question', '')} "
            f"{input_dict.get('description', '')} "
            f"{processed_conversation[-1]['text'] if len(processed_conversation) <= 2 else processed_conversation[0]['text'] + processed_conversation[-1]['text']}"
        )
        problem_list_manual, selected_doc_manual, retrieved_docs_manual = retrieve_docs_manual(
                question_category=question_category,
                category_mapping=ast.literal_eval(os.getenv('CATEGORY_MAPPING', '{}')),
                question_subcategory=input_dict.get('subcategory'),
                subcategory_mapping=ast.literal_eval(os.getenv('SUBCATEGORY_MAPPING', '{}')),
                question_info=question_info,
                get_prompt=prompts.get_choose_problem_path_prompt)
        logger.info('List of problems: %s', problem_list_manual)
        logger.info('Selected manual document: %s', selected_doc_manual)
        logger.info('Retrieved manual documents: %s', retrieved_docs_manual)

# Response generation
    response_0 = response = ''
    if question_category in assignment_categories:
        response_0 = generate(
            prompt=prompts.get_first_assignment_prompt(
                processed_conversation=processed_conversation,
                retrieved_qa_pairs=retrieved_qa_pairs,
                retrieved_docs_manual=retrieved_docs_manual
            )
        )
        logger.info('Initial response (assignment question): %s', response_0)
        response = generate(
            prompt=prompts.get_second_assignment_prompt(
                processed_conversation=processed_conversation,
                first_answer=response_0
            )
        )
    elif question_category in content_categories:
        response = generate(
            prompt=prompts.get_content_prompt(
                processed_conversation=processed_conversation,
                retrieved_qa_pairs=retrieved_qa_pairs,
                retrieved_docs_hybrid=retrieved_docs_hybrid
            )
        )
    elif question_category in logistics_categories:
        response = generate(
            prompt=prompts.get_logistics_prompt(
                processed_conversation=processed_conversation,
                retrieved_qa_pairs=retrieved_qa_pairs,
                retrieved_docs_hybrid=retrieved_docs_hybrid
            )
        )
    elif question_category in worksheet_categories:
        response = generate(
            prompt=prompts.get_worksheet_prompt(
                processed_conversation=processed_conversation,
                retrieved_qa_pairs=retrieved_qa_pairs,
                retrieved_docs_manual=retrieved_docs_manual,
                retrieved_docs_hybrid=retrieved_docs_hybrid
            )
         )
    logger.info('Final response: %s', response)
    
    # Logging and posting
    output_dict = {
        'processed_conversation': processed_conversation,
        'processed_conversation_search': processed_conversation_search,
        'retrieved_qa_pairs': retrieved_qa_pairs,
        'retrieved_docs_hybrid': retrieved_docs_hybrid,
        'problem_list_manual': problem_list_manual,
        'selected_doc_manual': selected_doc_manual,
        'retrieved_docs_manual': retrieved_docs_manual,
        'response_0': response_0,
        'response': response
    }
    
    # prod = input_dict['prod'] == 'true'
    # version = os.getenv('EDISON_VERSION')
    # experiment_name = input_dict.get('experiment_name', 'test')

    # if input_dict.get('log_blob') == 'true':
    #     log_path_blob = f"logs/{'production' if prod else 'test'}/{version if prod else experiment_name}.jsonl"
    #     log_blob({"inputs": input_dict, "outputs": output_dict}, log_path_blob)

    # if input_dict.get('log_local') == 'true':
    #     log_path_local = f"logs/{course}/{'production' if prod else 'test'}/{version if prod else experiment_name}.jsonl"
    #     log_local({"inputs": input_dict, "outputs": output_dict}, log_path_local)
    
    return jsonify({"Miloh": output_dict["response"]})



if __name__ == '__main__':
    app.run(debug=True)