import os
import json
import openai
import re
import time
from datetime import datetime
from typing import List, Tuple, Optional

# Configure OpenAI API (建议使用 .env 文件加载)
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")  # 可选，默认为空

# Prompt templates for Chinese and English
prompt_template_zh = """
你是一个精通各个学科的教师，帮助学生进行问题思路的指导而不是给出标准答案。请针对以下学科和难度级别自由生成一个问题，并给出思路指导，不给出答案。问题类型为{question_type}。
如果类型为简答题，对于特定的某些学科，必要的话给出代码和数学计算过程。不要返回多余的内容。

学科：{subject}
难度级别：{level}

以json格式返回：
"问题":""
"提供的思路":""
"""

prompt_template_en = """
You are an expert teacher in various subjects, helping students with problem-solving guidance instead of providing direct answers. Please freely generate a question and provide guidance for the given subject and difficulty level. The question type is {question_type}.
If the type is a short-answer question, include code or mathematical calculations where necessary for certain subjects. Do not include any extra content.
You should use English.

Subject: {subject}
Difficulty Level: {level}

Return in JSON format:
"Question": ""
"Guidance Provided": ""
"""


def send_request(prompt: str) -> Optional[str]:
    """Send a request to the OpenAI API and return the result"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048
        )

        if response.choices:
            response_text = response.choices[0].message['content']
            # Clean up the response by removing possible markdown code blocks
            cleaned_response = re.sub(r'```(json)?\s*|\s*```', '', response_text).strip()
            print(cleaned_response)
            return cleaned_response
    except Exception as e:
        print(f"API request failed: {e}")
        return None


def fix_json(response: str) -> Optional[str]:
    """Attempt to fix common JSON formatting errors"""
    try:
        # Try parsing directly
        json.loads(response)
        return response
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}, attempting to fix...")

    try:
        # Replace single quotes with double quotes
        fixed_response = response.replace("'", '"')
        json.loads(fixed_response)
        return fixed_response
    except json.JSONDecodeError:
        pass

    try:
        # Attempt to fix missing commas or colons
        fixed_response = re.sub(r'(?<=[}\]"\'\w])\s*(?=[{\["\'\w])', ', ', response)
        fixed_response = re.sub(r'(?<=[:])\s*(?=[{\["\'\w])', ' ', fixed_response)
        json.loads(fixed_response)
        return fixed_response
    except json.JSONDecodeError:
        pass

    print("Unable to fix JSON formatting errors")
    return None


def validate_response(response: str) -> bool:
    """Validate whether the API response is valid"""
    try:
        # Attempt to fix JSON format
        fixed_response = fix_json(response)
        if not fixed_response:
            return False

        data = json.loads(fixed_response)

        # Check if required fields exist
        required_keys = ["Question", "Guidance Provided"]
        if not all(key in data for key in required_keys):
            print("Validation failed: Missing required fields")
            return False

        # Check if field types are correct
        if not all(isinstance(data[key], str) for key in required_keys):
            print("Validation failed: Field type error")
            return False

        # Check if fields are empty strings
        if any(not data[key] for key in required_keys):
            print("Validation failed: Fields are empty")
            return False

        return True
    except Exception as e:
        print(f"Validation failed: JSON parsing error - {e}")
        return False


def get_question_and_answer(subject: str, level: str, question_type: str, lang: str = "en") -> Optional[dict]:
    """Get a question and its corresponding guidance based on language selection."""
    prompt_template = prompt_template_zh if lang == "zh" else prompt_template_en
    prompt = prompt_template.format(subject=subject, level=level, question_type=question_type)

    response = send_request(prompt)
    if not response or not validate_response(response):
        print(f"Generation failed: {subject}-{level}-{question_type}-{lang}")
        return None

    try:
        qa_data = json.loads(response)
        return {
            "Subject": subject,
            "Difficulty Level": level,
            "Question Type": question_type,
            "Language": lang,
            "Question": qa_data.get("Question", "") or qa_data.get("问题", ""),
            "Guidance Provided": qa_data.get("Guidance Provided", "") or qa_data.get("提供的思路", ""),
            "Generation Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        print(f"Parsing failed: Unable to extract data - {e}")
        return None


def process_subjects(subject_list: List[Tuple[str, str]], output_file: str, lang: str = "en"):
    """Process all subject and difficulty combinations, generating five questions per type."""
    question_types = ["Single Choice", "Multiple Choice", "Short Answer"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'a', encoding='utf-8') as outfile:
        for idx, (subject, level) in enumerate(subject_list, 1):
            print(f"Processing [{idx}/{len(subject_list)}] {subject}-{level}-{lang}")

            for q_type in question_types:
                successful_attempts = 0

                while successful_attempts < 5:
                    result = get_question_and_answer(subject, level, q_type, lang)

                    if result:
                        try:
                            result["Generation Index"] = successful_attempts + 1
                            result["Generation Time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                            outfile.flush()

                            successful_attempts += 1
                            print(f"Saved successfully: {subject}-{level}-{q_type}-{lang} ({successful_attempts}/5)")
                        except Exception as e:
                            print(f"File writing failed: {e}")
                    else:
                        print(f"Generation failed: {subject}-{level}-{q_type}-{lang}")

                    time.sleep(1)


def load_subject_list() -> List[Tuple[str, str]]:
    """Return a list of subjects and difficulty levels"""
    return [
        # Basic Education
        *[(subj, level) for subj in ["Chinese", "Mathematics", "English", "Physics", "Chemistry", "Biology", "History", "Geography"]
          for level in ["Elementary School", "Middle School", "High School"]],

        # Higher Education
        *[(subj, level) for subj in [
            "Mathematics", "Physics", "Chemistry", "Biology",
            "Computer Science", "Automation",
            "Aquaculture", "Crop Science",
            "Applied Economics", "Theoretical Economics",
            "General Pedagogy", "Physical Education",
            "Law",
            "Business Administration", "Public Administration",
            "Basic Medicine", "Clinical Medicine",
            "Sociology", "Literature and Art", "Psychology", "History", "Military Science"
        ] for level in ["Undergraduate", "Master", "PhD"]]
    ]


def main(lang: str = "en"):
    """Main function with optional language parameter."""
    output_dir = os.getenv("OUTPUT_DIR", "./data")
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"{lang}_helper_{current_time}.jsonl")

    subject_list = load_subject_list()

    process_subjects(subject_list, output_file, lang)

    print(f"Processing completed, results saved to: {output_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        lang = sys.argv[1].lower()
        if lang not in ("zh", "en"):
            print("Usage: python script.py [zh|en]")
        else:
            main(lang)
    else:
        main("en")  # 默认英文