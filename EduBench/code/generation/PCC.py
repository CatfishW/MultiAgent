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
你是一个个性化服务定制专家，能够基于学生画像，提供定制化的服务，提升学习效率，你需要实现：
1. 学习路径规划建议：基于学生当前的能力水平和学习目标【学生画像】，安排未来学习的课程顺序等。
2. 个性化意见生成：基于学生当前的知识薄弱点、学习习惯【学生画像】，生成或推荐适合难度的练习题或推荐阅读材料。

请针对以下学科和难度级别自由生成一个合适且具体的学生画像，并给出学习路径规划建议和个性化意见。

学科：{subject}

以json格式返回：
"学生画像":""
"学习路径规划建议":""
"个性化意见生成":""
"""

prompt_template_en = """
You are an expert in personalized service customization. Based on a student's profile, you need to provide tailored services to improve learning efficiency. This includes:
1. Learning Path Planning: Suggest a sequence of courses based on the student's current skill level and learning goals.
2. Personalized Recommendations: Generate or recommend exercises and reading materials suitable for the student's weak points and study habits.

Please generate an appropriate and specific student profile for the given subject and difficulty level, along with learning path planning suggestions and personalized recommendations.

Subject: {subject}

Return in JSON format:
"Student Profile": ""
"Learning Path Planning": ""
"Personalized Recommendations": ""
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

    print("Unable to fix JSON formatting error")
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
        required_keys = ["Student Profile", "Learning Path Planning", "Personalized Recommendations"]
        if not all(key in data for key in required_keys):
            print("Validation failed: Missing required fields")
            return False

        # Check if fields are empty
        if any(not data[key] for key in required_keys):
            print("Validation failed: Fields are empty")
            return False

        # Lenient validation: Accept dict, list, or string types
        if not isinstance(data["Student Profile"], (dict, list, str)):
            print(f"Validation failed: 'Student Profile' field type not supported - {type(data['Student Profile'])}")
            return False

        if not isinstance(data["Learning Path Planning"], (list, dict, str)):
            print(f"Validation failed: 'Learning Path Planning' field type not supported - {type(data['Learning Path Planning'])}")
            return False

        if not isinstance(data["Personalized Recommendations"], (dict, list, str)):
            print(f"Validation failed: 'Personalized Recommendations' field type not supported - {type(data['Personalized Recommendations'])}")
            return False

        return True
    except Exception as e:
        print(f"Validation failed: JSON parsing error - {e}")
        return False


def get_student_profile(subject: str, level: str, lang: str = "en") -> Optional[dict]:
    """Generate a student profile with learning plan and recommendations based on language selection."""
    prompt_template = prompt_template_zh if lang == "zh" else prompt_template_en
    prompt = prompt_template.format(subject=subject, level=level)

    response = send_request(prompt)
    if not response or not validate_response(response):
        print(f"Generation failed: {subject}-{level}-{lang}")
        return None

    try:
        profile_data = json.loads(response)

        def process_field(field_name):
            value = profile_data.get(field_name, "")
            if isinstance(value, (dict, list)):
                return value
            elif isinstance(value, str):
                return value.strip()
            else:
                return str(value).strip()

        return {
            "Subject": subject,
            "Education Level": level,
            "Language": lang,
            "Student Profile": process_field("Student Profile"),
            "Learning Path Planning": process_field("Learning Path Planning"),
            "Personalized Recommendations": process_field("Personalized Recommendations"),
            "Generation Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    except Exception as e:
        print(f"Parsing failed: Unable to extract data - {e}")
        return None


def process_subjects(subject_list: List[Tuple[str, str]], output_file: str, lang: str = "en"):
    """Process all subject and education levels, generating 5 entries per subject-level pair."""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'a', encoding='utf-8') as outfile:
        for idx, (subject, level) in enumerate(subject_list, 1):
            print(f"Processing [{idx}/{len(subject_list)}] {subject}-{level}-{lang}")

            successful_attempts = 0
            while successful_attempts < 5:
                result = get_student_profile(subject, level, lang)

                if result:
                    try:
                        result["Generation Index"] = successful_attempts + 1
                        result["Generation Time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                        outfile.flush()

                        successful_attempts += 1
                        print(f"Saved successfully: {subject}-{level}-{lang} ({successful_attempts}/5)")
                    except Exception as e:
                        print(f"File writing failed: {e}")
                else:
                    print(f"Generation failed: {subject}-{level}-{lang}")

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
    output_file = os.path.join(output_dir, f"{lang}_student_profile_{current_time}.jsonl")

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