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
import sys
from traceback import format_exc

# Ensure logs go to stdout (captured by Azure)
root = logging.getLogger()
if not root.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s :: %(message)s'))
    root.addHandler(h)
root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@app.errorhandler(Exception)
def _handle_unhandled(e):
    logger.error("UNHANDLED %s %s\n%s", request.method, request.path, format_exc())
    return jsonify(error="Internal Server Error", detail=str(e)), 500

@app.route("/health")
def health():
    return "ok", 200

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
    try:
        if request.headers.get('Authorization') != os.getenv('API_KEY'):
            logger.warning('Unauthorized access attempt')
            return jsonify(error='Unauthorized'), 401

        input_dict = request.get_json(silent=True) or {}
        logger.info('miloh: payload keys=%s', list(input_dict.keys()))

        # ---- load_course_config (common source of 500s if files/modules/env are missing)
        try:
            logger.info("miloh: loading course config 'ds100_miloh'")
            load_course_config('ds100_miloh')  # imports prompts.* and loads configs/ds100.env
            logger.info("miloh: course config loaded OK")
        except Exception:
            logger.error("miloh: load_course_config failed\n%s", format_exc())
            raise

        # ---- env categories
        try:
            assignment_categories = get_env_list('ASSIGNMENT_CATEGORIES')
            content_categories    = get_env_list('CONTENT_CATEGORIES')
            logistics_categories  = get_env_list('LOGISTICS_CATEGORIES')
            worksheet_categories  = get_env_list('WORKSHEET_CATEGORIES')
            logger.info("miloh: category sizes a=%d c=%d l=%d w=%d",
                        len(assignment_categories), len(content_categories),
                        len(logistics_categories), len(worksheet_categories))
        except Exception:
            logger.error("miloh: category env parse failed\n%s", format_exc())
            raise

        # ---- conversation preprocessing
        try:
            conversation_history = []
            initial_text = (
                f"Assignment: {input_dict.get('assignment','')}\n"
                f"Question: {input_dict.get('question','')}\n"
                f"Description: {input_dict.get('description','')}"
            )
            conversation_history.append({"user_role": "student", "text": initial_text, "document": None})
            for c in input_dict.get('chat', []):
                conversation_history.append({"user_role": "student", "text": c, "document": None})

            thread_title = f"{input_dict.get('assignment','')} — {input_dict.get('question','')}"
            processed_conversation = ocr_process_input(
                thread_title=thread_title,
                conversation_history=conversation_history,
            )
            logger.info("miloh: processed_conversation len=%d", len(processed_conversation))

            processed_conversation_search = process_conversation_search(
                processed_conversation=processed_conversation,
                prompt_summarize=prompts.get_summarize_conversation_prompt(processed_conversation[:-1])
            )
            logger.info('miloh: conversation summarized for search')
        except Exception:
            logger.error("miloh: preprocessing failed\n%s", format_exc())
            raise

        # ---- retrievals
        try:
            top_k = int(os.getenv('QA_TOP_K', '3'))
            retrieved_qa_pairs = retrieve_qa(conversation=processed_conversation_search, top_k=top_k)
            logger.info('miloh: retrieved_qa_pairs top_k=%d', top_k)
        except Exception:
            logger.error("miloh: retrieve_qa failed\n%s", format_exc())
            raise

        question_category = 'Homeworks'  # NOTE: verify this actually exists in env list

        try:
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
            logger.info('miloh: retrieved_docs_hybrid type=%s', type(retrieved_docs_hybrid).__name__)
        except Exception:
            logger.error("miloh: hybrid retrieval failed\n%s", format_exc())
            raise

        try:
            problem_list_manual = selected_doc_manual = retrieved_docs_manual = 'none'
            if question_category in (assignment_categories + worksheet_categories):
                question_info = re.sub(
                    r"\n+"," ",
                    f"{question_category} "
                    f"{input_dict.get('assignment', '')} "
                    f"{input_dict.get('question', '')} "
                    f"{input_dict.get('description', '')} "
                    f"{processed_conversation[-1]['text'] if processed_conversation else ''}"
                )
                out = retrieve_docs_manual(
                    question_category=question_category,
                    category_mapping=ast.literal_eval(os.getenv('CATEGORY_MAPPING', '{}')),
                    question_subcategory=input_dict.get('subcategory'),
                    subcategory_mapping=ast.literal_eval(os.getenv('SUBCATEGORY_MAPPING', '{}')),
                    question_info=question_info,
                    get_prompt=prompts.get_choose_problem_path_prompt
                )
                try:
                    problem_list_manual, selected_doc_manual, retrieved_docs_manual = out
                except Exception:
                    logger.error("miloh: retrieve_docs_manual returned: %r (expected 3-tuple)", out)
                    raise
            logger.info('miloh: manual retrieval done')
        except Exception:
            logger.error("miloh: manual retrieval failed\n%s", format_exc())
            raise

        # ---- generation
        try:
            response_0 = response = ''
            if question_category in assignment_categories:
                response_0 = generate(
                    prompt=prompts.get_first_assignment_prompt(
                        processed_conversation=processed_conversation,
                        retrieved_qa_pairs=retrieved_qa_pairs,
                        retrieved_docs_manual=retrieved_docs_manual
                    )
                )
                logger.info('miloh: first assignment response generated (len=%d)', len(response_0 or ""))

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
            logger.info('miloh: final response length=%d', len(response or ""))
        except Exception:
            logger.error("miloh: generation failed\n%s", format_exc())
            raise

        return jsonify({"Miloh": response})

    except Exception:
        logger.error("miloh: unhandled failure\n%s", format_exc())
        return jsonify(error="Internal Server Error"), 500

if __name__ == '__main__':
    app.run(debug=True)