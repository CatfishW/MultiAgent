import os
import json
import openai
import re
import time
from datetime import datetime
from typing import List, Tuple, Optional

# Configure OpenAI API (建议使用 .env 文件加载)
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE", "")

# Prompt templates for Chinese and English
prompt_template_zh = """
你是一个智能体，能够根据学生的个体差异，为每位学生生成个性化的学习内容或任务
请针对以下学科和学制，自由生成一段和正在学习{subject}的学生的画像，考虑下面的三个要点
1. 一对一：基于特定的学生画像，定制专属练习题或阅读材料；
2. 分层教学：针对相同的课程内容，对不同层次的同学生成不同的教学目标、教学方法、学习结果检测方式、课下作业布置等；
3. 其他：结合其他的能力需求，考虑单一学生、学习小组、班级三个层次区分不同的能力评测数据设计

不要返回多余的内容。

学制级别：{level}

严格按照json格式返回
"学生的画像":""
"个性化学习内容/任务":""
"""

prompt_template_en = """
You are an intelligent agent capable of generating personalized learning content or tasks for each student based on their individual differences.
For the following subject and educational level, freely create a profile of a student currently studying {subject}, considering the three key points below:

One-on-one: Customize exclusive exercises or reading materials based on a specific student profile;
Tiered teaching: For the same course content, generate different teaching objectives, methods, learning outcome assessment approaches, and homework assignments for students at different levels;
Other: Considering other competency requirements, design differentiated ability evaluation data for individual students, study groups, and entire classes.
Do not include any extra content.

Educational level: {level}

Strictly return in JSON format:
"Student Profile": ""
"Personalized Learning Content/Task": ""
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
        required_keys = ["Student Profile", "Personalized Learning Content/Task"]
        if not all(key in data for key in required_keys):
            print(f"Validation failed: Missing required fields. Required fields: {required_keys}")
            return False

        # Check 'Student Profile' type and non-emptiness
        student_profile = data["Student Profile"]
        if isinstance(student_profile, str):
            if not student_profile.strip():
                print("Validation failed: Field 'Student Profile' is an empty string")
                return False
        elif isinstance(student_profile, dict):
            if not student_profile:
                print("Validation failed: Field 'Student Profile' is an empty dictionary")
                return False
        else:
            print("Validation failed: Field 'Student Profile' has incorrect type, should be a string or dictionary")
            return False

        # Check 'Personalized Learning Content/Task' type and content
        personalized_content = data["Personalized Learning Content/Task"]
        if not isinstance(personalized_content, dict):
            print("Validation failed: Field 'Personalized Learning Content/Task' has incorrect type, should be a dictionary")
            return False

        # Define required nested keys (case-insensitive)
        nested_required_keys = ["One-on-one", "Tiered Teaching", "Other"]

        # Normalize keys to lowercase for comparison
        normalized_personalized_content = {key.lower(): value for key, value in personalized_content.items()}

        # Validate nested fields
        for key in nested_required_keys:
            lower_key = key.lower()
            if lower_key not in normalized_personalized_content:
                print(f"Validation failed: Missing nested field. Required field: {key}")
                return False

            nested_value = normalized_personalized_content[lower_key]
            if not isinstance(nested_value, dict):
                print(f"Validation failed: Field '{key}' has incorrect type, should be a dictionary")
                return False
            if not nested_value:
                print(f"Validation failed: Field '{key}' is empty")
                return False

        return True

    except json.JSONDecodeError as e:
        print(f"Validation failed: JSON parsing error - {e}")
        return False
    except Exception as e:
        print(f"Validation failed: Unknown error - {e}")
        return False


def get_question_and_answer(subject: str, level: str, lang: str = "en") -> Optional[dict]:
    """Get question and answer based on language selection."""
    prompt_template = prompt_template_zh if lang == "zh" else prompt_template_en
    prompt = prompt_template.format(subject=subject, level=level)

    response = send_request(prompt)
    if not response or not validate_response(response):
        print(f"Generation failed: {subject}-{level}-{lang}")
        return None

    # Parse JSON data
    try:
        qa_data = json.loads(response)
        return {
            "Subject": subject,
            "Education Level": level,
            "Language": lang,
            "Student Profile": qa_data["Student Profile"],
            "Personalized Learning Content/Task": qa_data["Personalized Learning Content/Task"],  # Could be a string or nested dictionary
            "Generation Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        print(f"Parsing failed: Unable to extract data - {e}")
        return None


def process_subjects(subject_list: List[Tuple[str, str]], output_file: str, lang: str = "en"):
    """Process all subject and difficulty combinations, generating one successful result for each question."""

    with open(output_file, 'a', encoding='utf-8') as outfile:
        for idx, (subject, level) in enumerate(subject_list, 1):
            print(f"Processing [{idx}/{len(subject_list)}] {subject}-{level}-{lang}")

            result = get_question_and_answer(subject, level, lang)
            if result:
                try:
                    # Write to file
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    outfile.flush()
                    print(f"Saved successfully: {subject}-{level}-{lang}")
                except Exception as e:
                    print(f"File writing failed: {e}")


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
    output_file = os.path.join(output_dir, f"{lang}_design_{current_time}.jsonl")

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