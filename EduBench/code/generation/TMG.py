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
你负责帮忙教师进行教学素材生成，根据教材章节或知识点，自动生成结构化教案，包括教学目标、重点难点、课堂活动设计等；
请针对以下学科和难度级别自由生成一个合适的知识点，自动生成结构化教案，包括教学目标、重点难点、课堂活动设计等。问题类型为{question_type}。
如果类型为简答题，对于特定的某些学科，必要的话给出代码和数学计算过程。不要返回多余的内容。

学科：{subject}
难度级别：{level}

以json格式返回
"知识点":""
"教学素材":""
"""

prompt_template_en = """
You are responsible for helping teachers generate teaching materials. Based on the textbook chapter or knowledge point, automatically generate a structured lesson plan including learning objectives, key points and difficulties, and classroom activity design.
Please freely generate an appropriate knowledge point for the given subject and difficulty level, and automatically create a structured lesson plan including learning objectives, key points and difficulties, and classroom activity design. The question type is {question_type}.
If the type is a short-answer question, include code or mathematical calculations where necessary for certain subjects. Do not include any extra content.

Subject: {subject}
Difficulty Level: {level}

Return in JSON format:
"Knowledge Point": ""
"Teaching Materials": ""
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
        required_keys = ["Knowledge Point", "Teaching Materials"]
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
    """Get a knowledge point and its teaching materials based on language selection."""
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
            "Level": level,
            "Question Type": question_type,
            "Language": lang,
            "Knowledge Point": qa_data.get("Knowledge Point", "") or qa_data.get("知识点", ""),
            "Teaching Materials": qa_data.get("Teaching Materials", "") or qa_data.get("教学素材", ""),
            "Generation Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        print(f"Parsing failed: Unable to extract data - {e}")
        return None


def load_processed_combinations(output_file: str) -> set:
    """Load already processed combinations from the output file."""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    data = json.loads(line.strip())
                    combination = (data["Subject"], data["Level"], data["Question Type"])
                    processed.add(combination)
                except Exception as e:
                    print(f"Failed to parse line in progress file: {e}")
    return processed


def process_subjects(subject_list: List[Tuple[str, str]], output_file: str, lang: str = "en"):
    """Process all subject and difficulty combinations, generating five results per question type."""
    question_types = ["Single Choice", "Multiple Choice", "Short Answer"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    processed_combinations = load_processed_combinations(output_file)

    with open(output_file, 'a', encoding='utf-8') as outfile:
        for idx, (subject, level) in enumerate(subject_list, 1):
            print(f"Processing [{idx}/{len(subject_list)}] {subject}-{level}-{lang}")

            for q_type in question_types:
                combination = (subject, level, q_type)

                if combination in processed_combinations:
                    print(f"Skipping already processed: {combination}")
                    continue

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
                            processed_combinations.add(combination)
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
    output_file = os.path.join(output_dir, f"{lang}_material_{current_time}.jsonl")

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