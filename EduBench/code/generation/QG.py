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
你是一个精通各个学科的教师，根据知识点范围和题型来给出合适的题目。请针对以下学科和难度级别自由生成一个知识点和对应的问题，并给出思路指导，并给出答案。问题类型为{question_type}。
如果类型为简答题，对于特定的某些学科，必要的话给出代码和数学计算过程。不要返回多余的内容。

学科：{subject}
难度级别：{level}

以json格式返回
"知识点":""
"问题":""
"提供的思路":""
"答案":""
"""

prompt_template_en = """
You are a teacher proficient in various subjects. Based on the subject and difficulty level, generate an appropriate knowledge point, question, solution guidance, and answer. The question type is {question_type}.
If the type is a short-answer question, include code or mathematical calculations where necessary for certain subjects. Do not include any extra content.

Subject: {subject}
Difficulty Level: {level}

Return in JSON format:
"Knowledge Point": ""
"Question": ""
"Solution Guidance": ""
"Answer": ""
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


def validate_response(response: str) -> bool:
    """Validate whether the API response is valid"""
    try:
        data = json.loads(response)

        # Check if required fields exist
        required_keys = ["Knowledge Point", "Question", "Solution Guidance", "Answer"]
        if not all(key in data for key in required_keys):
            print(f"Validation failed: Missing required fields. Required fields: {required_keys}")
            return False

        # Check field content is not empty or invalid
        for key in required_keys:
            field_content = str(data[key]).strip()  # Convert to string and remove whitespace
            if not field_content:  # Check if empty
                print(f"Validation failed: Field '{key}' is empty or invalid")
                return False

        return True

    except json.JSONDecodeError as e:
        print(f"Validation failed: JSON parsing error - {e}")
        return False
    except Exception as e:
        print(f"Validation failed: Unknown error - {e}")
        return False


def get_question_and_answer(subject: str, level: str, question_type: str, lang: str = "en") -> Optional[dict]:
    """Generate a question with knowledge point, guidance, and answer based on language selection."""
    prompt_template = prompt_template_zh if lang == "zh" else prompt_template_en
    prompt = prompt_template.format(subject=subject, level=level, question_type=question_type)

    response = send_request(prompt)
    if not response or not validate_response(response):
        print(f"Generation failed: {subject}-{level}-{question_type}-{lang}")
        return None

    try:
        qa_data = json.loads(response)

        def process_field(field_name):
            value = qa_data.get(field_name, "")
            if isinstance(value, list):
                return " ".join(str(item).strip() for item in value)
            elif isinstance(value, str):
                return value.strip()
            else:
                return str(value).strip()

        result = {
            "Subject": subject,
            "Education Level": level,
            "Question Type": question_type,
            "Language": lang,
            "Knowledge Point": process_field("Knowledge Point"),
            "Question": process_field("Question"),
            "Solution Guidance": process_field("Solution Guidance"),
            "Answer": process_field("Answer"),
            "Generation Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return result

    except Exception as e:
        print(f"Parsing failed: Unable to extract data - {e}")
        return None


def process_subjects(subject_list: List[Tuple[str, str]], output_file: str, lang: str = "en"):
    """Process all subject and education levels, generating 5 entries per question type."""
    question_types = ["Single Choice", "Multiple Choice", "Short Answer"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'a', encoding='utf-8') as outfile:
        for idx, (subject, level) in enumerate(subject_list, 1):
            print(f"Processing [{idx}/{len(subject_list)}] {subject}-{level}-{lang}")

            for q_type in question_types:
                for repeat in range(5):  # Generate 5 samples per question type
                    result = get_question_and_answer(subject, level, q_type, lang)

                    if result:
                        try:
                            # Only write non-empty fields
                            valid_result = {k: v for k, v in result.items() if v}
                            outfile.write(json.dumps(valid_result, ensure_ascii=False) + '\n')
                            outfile.flush()

                            print(f"Saved successfully: {subject}-{level}-{q_type}-{lang} ({repeat + 1}/5)")
                        except Exception as e:
                            print(f"File writing failed: {e}")
                    else:
                        print(f"Generation failed: {subject}-{level}-{q_type}-{lang} (Attempt {repeat + 1})")

                    time.sleep(1)


def load_subject_list() -> List[Tuple[str, str]]:
    """Return a list of subjects and education levels"""
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
    output_file = os.path.join(output_dir, f"{lang}_question_gen_{current_time}.jsonl")

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
        main("en")  # Default to English