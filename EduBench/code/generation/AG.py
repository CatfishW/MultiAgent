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
你需要实现：
1. 客观打分能力：（选择、判断、填空）；对给出步骤分数或者打分参考的解答题进行打分；
2. 主观打分能力：对课程大作业、实验报告等从不同维度，例如工作量、完整性、知识运用程度等维度进行综合评估打分；
3. 个性化反馈能力：可以针对学生答题情况生成具体、有建设性的反馈意见，例如可能涉及的知识盲区，学习建议等；【打分+建议】

请针对以下学科和难度级别自由生成一个合适的问题和学生的回答，针对学生的答案给出评分。问题类型为{question_type}。
如果类型为简答题，对于特定的某些学科，必要的话给出代码和数学计算过程。不要返回多余的内容。

学科：{subject}
难度级别：{level}

以json格式返回
"问题":""
"学生的答案":""
"评分":""
"评分细节":""
"个性化反馈":""
"""

prompt_template_en = """
You need to implement:
1. Objective scoring capability (e.g., multiple-choice, true/false, fill-in-the-blank); provide step-by-step scoring or grading references for problem-solving questions.
2. Subjective scoring capability: Evaluate comprehensive assignments, lab reports, etc., based on various dimensions such as workload, completeness, and knowledge application.
3. Personalized feedback capability: Generate specific and constructive feedback for students' answers, including potential knowledge gaps and learning suggestions.

Please freely generate an appropriate question and a student's answer for the given subject and difficulty level. Provide a score for the student's answer. The question type is {question_type}.
If the type is a short-answer question, include code or mathematical calculations where necessary for certain subjects. Do not include any extra content.
You should use English.

Subject: {subject}
Difficulty Level: {level}

Return in JSON format:
"Question": ""
"Student's Answer": ""
"Score": ""
"Scoring Details": ""
"Personalized Feedback": ""
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
        required_keys = ["Question", "Student's Answer", "Score", "Scoring Details", "Personalized Feedback"]
        if not all(key in data for key in required_keys):
            print(f"Validation failed: Missing required fields. Required fields: {required_keys}")
            return False

        # Check field content is not empty or invalid
        for key in required_keys:
            if not str(data[key]).strip():  # Convert to string and check if empty
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
    """Get a question and its corresponding answer with score and feedback based on language selection."""
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
            "Question": qa_data.get("Question", "") or qa_data.get("问题", ""),
            "Student's Answer": qa_data.get("Student's Answer", "") or qa_data.get("学生的答案", ""),
            "Score": qa_data.get("Score", "") or qa_data.get("评分", ""),
            "Scoring Details": qa_data.get("Scoring Details", "") or qa_data.get("评分细节", ""),
            "Personalized Feedback": qa_data.get("Personalized Feedback", "") or qa_data.get("个性化反馈", ""),
            "Generation Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        print(f"Parsing failed: Unable to extract data - {e}")
        return None


def process_subjects(subject_list: List[Tuple[str, str]], output_file: str, lang: str = "en"):
    """Process all subject and difficulty combinations, generating four results per question type."""
    question_types = ["Multiple Choice", "True/False", "Short Answer"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'a', encoding='utf-8') as outfile:
        for idx, (subject, level) in enumerate(subject_list, 1):
            print(f"Processing [{idx}/{len(subject_list)}] {subject}-{level}-{lang}")

            for q_type in question_types:
                for repeat in range(4):  # Generate 4 samples per type
                    result = get_question_and_answer(subject, level, q_type, lang)

                    if result:
                        try:
                            result["Generation Index"] = repeat + 1
                            result["Generation Time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                            outfile.flush()

                            print(f"Saved successfully: {subject}-{level}-{q_type}-{lang} ({repeat + 1}/4)")
                        except Exception as e:
                            print(f"File writing failed: {e}")
                    else:
                        print(f"Generation failed: {subject}-{level}-{q_type}-{lang} (Attempt {repeat + 1})")

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
    output_file = os.path.join(output_dir, f"{lang}_judge_{current_time}.jsonl")

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
