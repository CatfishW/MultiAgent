from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
import time
from tqdm import tqdm

# Simulated model functions
def get_normal_answer(prompt, model_name):
    """Simulate calling a model and returning a result"""
    print(f"Calling model: {model_name}")
    time.sleep(1)  # Simulate delay
    return f"Answer from {model_name}"


def get_reasoning_answer(prompt, model_name):
    """Simulate calling a reasoning model"""
    reasoning = f"Reasoning steps for {model_name}"
    answer = f"Final answer from {model_name}"
    return reasoning, answer


# Global lock to ensure safe file writing
write_lock = threading.Lock()

# Supported task types and their prompt templates
TASK_PROMPT_TEMPLATES = {
    "QA": lambda x: f"{x}\nPlease generate the corresponding answer.\nReturn in JSON format:\n\"Answer\": ",
    "EC": lambda q, a: f"{q}\nStudent's original answer: {a}\nProvide corrected answer and explanation.\nReturn in JSON format:\n\"Corrected Answer\": \n\"Error Explanation\": ",
    "IP": lambda q: f"{q}\nPlease provide a step-by-step solution approach.\nReturn in JSON format:\n\"Provided Approach\": ",
    "PLS": lambda sp: f"Based on this student profile:\n{sp}\nGenerate learning path and recommendations.\nReturn in JSON format:\n\"Learning Path Planning\": \n\"Personalized Recommendations\": ",
    "ES": lambda c, l: f"Dialogue with student:\n{c}\nAnxiety level: {l}\nProvide emotional analysis and comfort suggestions.\nReturn in JSON format:\n\"Emotional State Analysis\": \n\"Comfort & Suggestions\": ",
    "QG": lambda kp, subj, qt, lv: f"Knowledge Point: {kp}\nSubject: {subj}\nDifficulty Level: {lv}\nQuestion Type: {qt}\nGenerate a question.\nReturn in JSON format:\n\"Question\": ",
    "AG": lambda q, sa: f"Question: {q}\nStudent Answer: {sa}\nProvide score, scoring details, and feedback.\nReturn in JSON format:\n\"Score\": \n\"Scoring Details\": \n\"Personalized Feedback\": ",
    "TMG": lambda kp: f"Knowledge Point: {kp}\nGenerate teaching material including objectives, key points, and activities.\nReturn in JSON format:\n\"Teaching Material\": ",
    "PCC": lambda sp: f"Student Profile: {sp}\nGenerate personalized learning content or task.\nReturn in JSON format:\n\"personalized learning content or task\": "
}


def process_single_model(args):
    """Thread function to call a single model"""
    model_type, model_name, prompt = args
    try:
        if model_type == 'normal':
            response = get_normal_answer(prompt, model_name)
            return {'model': model_name, 'reasoning': None, 'response': response}
        elif model_type == 'reasoning':
            reasoning, answer = get_reasoning_answer(prompt, model_name)
            return {'model': model_name, 'reasoning': reasoning, 'response': answer}
    except Exception as e:
        print(f"Model {model_name} call failed: {e}")
        return None


def process_single_task(task_id, task_type, input_data, normal_models, reasoning_models, output_file):
    """
    Process a single task based on its type and input data.
    """

    # Build prompt based on task type
    if task_type == "QA":
        prompt_content = TASK_PROMPT_TEMPLATES["QA"](input_data.get("question", ""))
    elif task_type == "EC":
        prompt_content = TASK_PROMPT_TEMPLATES["EC"](input_data.get("question", ""), input_data.get("original_answer", ""))
    elif task_type == "IP":
        prompt_content = TASK_PROMPT_TEMPLATES["IP"](input_data.get("question", ""))
    elif task_type == "PLS":
        prompt_content = TASK_PROMPT_TEMPLATES["PLS"](input_data.get("student_profile", ""))
    elif task_type == "ES":
        prompt_content = TASK_PROMPT_TEMPLATES["ES"](input_data.get("conversation", ""), input_data.get("anxiety_level", ""))
    elif task_type == "QG":
        prompt_content = TASK_PROMPT_TEMPLATES["QG"](
            input_data.get("knowledge_point", ""),
            input_data.get("subject", ""),
            input_data.get("question_type", ""),
            input_data.get("level", "")
        )
    elif task_type == "AG":
        prompt_content = TASK_PROMPT_TEMPLATES["AG"](input_data.get("question", ""), input_data.get("student_answer", ""))
    elif task_type == "TMG":
        prompt_content = TASK_PROMPT_TEMPLATES["TMG"](input_data.get("knowledge_point", ""))
    elif task_type == "PCC":
        prompt_content = TASK_PROMPT_TEMPLATES["PCC"](input_data.get("student_profile", ""))
    else:
        print(f"Unknown task type: {task_type}")
        return

    results = []
    tasks = []
    for model in normal_models:
        tasks.append(('normal', model, prompt_content))
    # for model in reasoning_models:
    #     tasks.append(('reasoning', model, prompt_content))

    with ThreadPoolExecutor(max_workers=5) as model_executor:
        futures = [model_executor.submit(process_single_model, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # Construct output
    output = {
        "task_id": task_id,
        "task_type": task_type,
        "input": input_data,
        "results": results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Write to file safely
    with write_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(output, ensure_ascii=False) + '\n')

    return task_id


def main():
    # Configuration
    input_jsonl_file = ""
    output_jsonl_file = ""

    normal_models = ['qwen2.5-7b-instruct', 'qwen2.5-14b-instruct', 'deepseek-v3']
    reasoning_models = ['deepseek-r1']

    # Load tasks
    tasks = []
    with open(input_jsonl_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                task_type = data.get("task_type")
                if not task_type:
                    print(f"Skipping task {idx}: missing 'task_type'")
                    continue
                tasks.append((idx, task_type, data))
            except json.JSONDecodeError as e:
                print(f"Failed to parse line {idx}: {e}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_jsonl_file), exist_ok=True)

    # Submit all tasks
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                process_single_task,
                tid,
                ttype,
                tdata,
                normal_models,
                reasoning_models,
                output_jsonl_file
            ) for tid, ttype, tdata in tasks
        ]

        with tqdm(total=len(tasks), desc="Processing Tasks") as pbar:
            for future in as_completed(futures):
                _ = future.result()  # Optional: retrieve result
                pbar.update(1)

    print(f"All tasks completed. Results saved to {output_jsonl_file}")


if __name__ == "__main__":
    main()
