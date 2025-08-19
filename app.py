import os
import re
import ast
import logging
from typing import Dict, Any
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from traceback import format_exc

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

# Make INFO logs visible (your previous basicConfig dropped INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = Flask(__name__)

load_dotenv('./keys.env')

def load_course_config(course: str) -> None:
    try:
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
        logger.info("load_course_config: loaded prompts and env for %s", course)
    except Exception:
        logger.error("load_course_config failed for course=%s\n%s", course, format_exc())
        raise

def get_env_list(key: str) -> list:
    try:
        val = os.getenv(key, '[]')
        lst = ast.literal_eval(val)
        logger.info("get_env_list: %s -> list(len=%s)", key, len(lst) if hasattr(lst, '__len__') else 'n/a')
        return lst
    except Exception:
        logger.error("get_env_list failed for key=%s (value=%r)\n%s", key, os.getenv(key), format_exc())
        raise

# Global error handler to surface Python tracebacks to logs and client
@app.errorhandler(Exception)
def _unhandled(e):
    logger.error("UNHANDLED %s %s\n%s", request.method, request.path, format_exc())
    return jsonify(error="Internal Server Error", detail=str(e)), 500

# Miloh Office Hours Extension
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
    try:
        if request.headers.get('Authorization') != os.getenv('API_KEY'):
            logger.warning('Unauthorized access attempt (Authorization header present: %s)',
                           'yes' if request.headers.get('Authorization') else 'no')
            return jsonify(error='Unauthorized'), 401

        # Get input data and load the course config
        try:
            input_dict = request.json or {}
        except Exception:
            logger.error("Failed to parse JSON body\n%s", format_exc())
            raise
        logger.info('Received input keys: %s', list(input_dict.keys()))

        course = 'ds100_miloh'
        try:
            load_course_config('ds100_miloh')
        except Exception:
            logger.error("miloh: load_course_config crashed\n%s", format_exc())
            raise

        try:
            assignment_categories = get_env_list('ASSIGNMENT_CATEGORIES')
            content_categories = get_env_list('CONTENT_CATEGORIES')
            logistics_categories = get_env_list('LOGISTICS_CATEGORIES')
            worksheet_categories = get_env_list('WORKSHEET_CATEGORIES')
        except Exception:
            logger.error("miloh: loading env category lists crashed\n%s", format_exc())
            raise

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
        try:
            for c in input_dict.get('chat', []):
                conversation_history.append({
                    "user_role": "student",
                    "text": c,
                    "document": None
                })
        except Exception:
            logger.error("miloh: building conversation_history crashed\n%s", format_exc())
            raise

        # For thread_title, combine assignment + question:
        thread_title = f"{input_dict.get('assignment','')} — {input_dict.get('question','')}"
        try:
            processed_conversation = ocr_process_input(
                thread_title=thread_title,
                conversation_history=conversation_history,
            )
            logger.info("Processed conversation length: %s", len(processed_conversation) if processed_conversation else 0)
        except Exception:
            logger.error("miloh: ocr_process_input crashed (thread_title=%r)\n%s", thread_title, format_exc())
            raise

        try:
            processed_conversation_search = process_conversation_search(
                processed_conversation=processed_conversation,
                prompt_summarize=prompts.get_summarize_conversation_prompt(processed_conversation[:-1])
            )
            logger.info('Processed (summarized) conversation for search: %s',
                        (processed_conversation_search[:200] + '...') if isinstance(processed_conversation_search, str) and len(processed_conversation_search) > 200 else processed_conversation_search)
        except Exception:
            logger.error("miloh: process_conversation_search crashed\n%s", format_exc())
            raise

        # QA retrieval
        try:
            top_k = int(os.getenv('QA_TOP_K', '3'))
        except Exception:
            logger.error("miloh: invalid QA_TOP_K=%r\n%s", os.getenv('QA_TOP_K'), format_exc())
            raise

        try:
            retrieved_qa_pairs = retrieve_qa(conversation=processed_conversation_search, top_k=top_k)
            logger.info('Retrieved QA pairs type=%s', type(retrieved_qa_pairs).__name__)
        except Exception:
            logger.error("miloh: retrieve_qa crashed (top_k=%s)\n%s", top_k, format_exc())
            raise

        question_category = 'Homeworks'
        logger.info("Question category: %s (in assignment=%s, content=%s, logistics=%s, worksheet=%s)",
                    question_category,
                    question_category in assignment_categories,
                    question_category in content_categories,
                    question_category in logistics_categories,
                    question_category in worksheet_categories)

        # Hybrid document retrieval
        retrieved_docs_hybrid = 'none'
        try:
            if question_category in content_categories:
                idx = os.getenv('CONTENT_INDEX_NAME')
                retrieved_docs_hybrid = retrieve_docs_hybrid(
                    text=processed_conversation_search,
                    index_name=idx,
                    top_k=int(os.getenv('CONTENT_INDEX_TOP_K', '1')),
                    semantic_reranking=True
                )
                logger.info('Hybrid retrieval (content) index=%r', idx)
            elif question_category in logistics_categories:
                idx = os.getenv('LOGISTICS_INDEX_NAME')
                retrieved_docs_hybrid = retrieve_docs_hybrid(
                    text=processed_conversation_search,
                    index_name=idx,
                    top_k=int(os.getenv('LOGISTICS_INDEX_TOP_K', '1')),
                    semantic_reranking=False
                )
                logger.info('Hybrid retrieval (logistics) index=%r', idx)
            elif question_category in worksheet_categories:
                idx = os.getenv('WORKSHEET_INDEX_NAME')
                retrieved_docs_hybrid = retrieve_docs_hybrid(
                    text=processed_conversation_search,
                    index_name=idx,
                    top_k=int(os.getenv('WORKSHEET_INDEX_TOP_K', '1')),
                    semantic_reranking=True
                )
                logger.info('Hybrid retrieval (worksheet) index=%r', idx)
            logger.info('Retrieved hybrid documents type=%s', type(retrieved_docs_hybrid).__name__)
        except Exception:
            logger.error("miloh: retrieve_docs_hybrid crashed\n%s", format_exc())
            raise

        # Manual document retrieval
        problem_list_manual = selected_doc_manual = retrieved_docs_manual = 'none'
        try:
            if question_category in (assignment_categories + worksheet_categories):
                try:
                    question_info = re.sub(
                        r"\n+",
                        " ",
                        f"{question_category} "
                        f"{input_dict.get('assignment', '')} "
                        f"{input_dict.get('question', '')} "
                        f"{input_dict.get('description', '')} "
                        f"{processed_conversation[-1]['text'] if len(processed_conversation) <= 2 else processed_conversation[0]['text'] + processed_conversation[-1]['text']}"
                    )
                except Exception:
                    logger.error("miloh: building question_info crashed\n%s", format_exc())
                    raise

                try:
                    problem_list_manual, selected_doc_manual, retrieved_docs_manual = retrieve_docs_manual(
                        question_category=question_category,
                        category_mapping=ast.literal_eval(os.getenv('CATEGORY_MAPPING', '{}')),
                        question_subcategory=input_dict.get('subcategory'),
                        subcategory_mapping=ast.literal_eval(os.getenv('SUBCATEGORY_MAPPING', '{}')),
                        question_info=question_info,
                        get_prompt=prompts.get_choose_problem_path_prompt)
                except Exception:
                    logger.error("miloh: retrieve_docs_manual crashed\n%s", format_exc())
                    raise

                logger.info('List of problems: %s', problem_list_manual)
                logger.info('Selected manual document: %s', selected_doc_manual)
                logger.info('Retrieved manual documents: %s', retrieved_docs_manual)
        except Exception:
            logger.error("miloh: manual retrieval block crashed\n%s", format_exc())
            raise

        # Response generation
        try:
            response_0 = response = ''
            if question_category in assignment_categories:
                try:
                    response_0 = generate(
                        prompt=prompts.get_first_assignment_prompt(
                            processed_conversation=processed_conversation,
                            retrieved_qa_pairs=retrieved_qa_pairs,
                            retrieved_docs_manual=retrieved_docs_manual
                        )
                    )
                    logger.info('Initial response (assignment question) length=%s', len(response_0 or ''))
                except Exception:
                    logger.error("miloh: first assignment generate crashed\n%s", format_exc())
                    raise

                try:
                    response = generate(
                        prompt=prompts.get_second_assignment_prompt(
                            processed_conversation=processed_conversation,
                            first_answer=response_0
                        )
                    )
                except Exception:
                    logger.error("miloh: second assignment generate crashed\n%s", format_exc())
                    raise
            elif question_category in content_categories:
                try:
                    response = generate(
                        prompt=prompts.get_content_prompt(
                            processed_conversation=processed_conversation,
                            retrieved_qa_pairs=retrieved_qa_pairs,
                            retrieved_docs_hybrid=retrieved_docs_hybrid
                        )
                    )
                except Exception:
                    logger.error("miloh: content generate crashed\n%s", format_exc())
                    raise
            elif question_category in logistics_categories:
                try:
                    response = generate(
                        prompt=prompts.get_logistics_prompt(
                            processed_conversation=processed_conversation,
                            retrieved_qa_pairs=retrieved_qa_pairs,
                            retrieved_docs_hybrid=retrieved_docs_hybrid
                        )
                    )
                except Exception:
                    logger.error("miloh: logistics generate crashed\n%s", format_exc())
                    raise
            elif question_category in worksheet_categories:
                try:
                    response = generate(
                        prompt=prompts.get_worksheet_prompt(
                            processed_conversation=processed_conversation,
                            retrieved_qa_pairs=retrieved_qa_pairs,
                            retrieved_docs_manual=retrieved_docs_manual,
                            retrieved_docs_hybrid=retrieved_docs_hybrid
                        )
                    )
                except Exception:
                    logger.error("miloh: worksheet generate crashed\n%s", format_exc())
                    raise
            logger.info('Final response length=%s', len(response or ''))
        except Exception:
            logger.error("miloh: response generation block crashed\n%s", format_exc())
            raise

        # Logging and posting (no logic change)
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

        return jsonify({"Miloh": output_dict["response"]})

    except Exception:
        # This catches anything missed above; global handler will also log.
        logger.error("miloh: unhandled exception at top level\n%s", format_exc())
        raise


if __name__ == '__main__':
    app.run(debug=True)