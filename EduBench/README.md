---
license: mit
language:
- zh
- en
---
# EduBench
here is the data repo for [EduBench](https://arxiv.org/abs/2505.16160)
## 1. Evaluation Scenarios
**I. Student-Oriented Scenarios**

- Question Answering (Q&A)
    - The ability of an AI system to accurately solve questions posed by students across
various subjects and difficulty levels. 
- Error Correction (EC)
    - The capacity to identify and correct student errors in assignments, exams, or
daily exercises. Errors can range from obvious mistakes to subtle issues such as variable misuse
in code or logical flaws in mathematical reasoning.
- Idea Provision (IP)
    - This includes answering student queries about knowledge points, homework guidance, or exam preparation. It is subdivided into basic factual explanations, step-by-step solution
analysis, and general academic advice.
- Personalized Learning Support (PLS) 
    - Based on student profiles (e.g., skill level, learning goals),
the system recommends learning paths, exercises, or reading materials tailored to individual needs.
- Emotional Support (ES) 
    - This involves detecting a student’s emotional state (e.g., anxiety before exams)
from text and offering appropriate supportive feedback or suggestions. Scenarios include pre-exam
stress, post-exam frustration, or social isolation.

**II. Teacher-Oriented Scenarios**

- Question Generation (QG)
    - : Generating questions based on specified topics, difficulty levels, and knowledge scopes. This includes both single-topic and multi-topic (comprehensive) question generation.
Advanced requirements involve generating explanations and formatting full exams. 
- Automatic Grading (AG) 
    - Supporting grading of objective questions (e.g., multiple-choice, fill-in-theblank) and subjective tasks (e.g., project reports) based on scoring rubrics. Feedback generation is
also supported. Metrics include scoring accuracy, reasonableness, and feedback informativeness.
- Teaching Material Generation (TMG)  
    - Automatically generating educational content such as slides,
teaching plans, and lecture notes. This includes content structuring and supplementing with relevant
external materials like images or references.
- Personalized Content Creation (PCC)  
    - Generating differentiated content for students based on their
learning levels or personal profiles. This includes both individualized assignments and tiered content
design (e.g., differentiated learning objectives, teaching strategies, and assessments for varying
student levels).

## 2. Statistics of EduBench
<div align="center">
  <img src="data_statistics.png" alt="Framework" width="1200"/>
  <br>
</div>

## 3. Data Format
Each JSONL file contains the following key fields:
- `information`: Metadata describing scenario attributes, such as subject domain and task difficulty level.

- `prompt`: The input text prompt used for model evaluation

- `model_predictions`: System responses from multiple LLMs, specifically including qwen2.5-7b-instruct, qwen2.5-14b-instruct, qwen-max, deepseek-v3, and deepseek-r1.

## 4. Human Annotated Data
If you need to obtain the human-annotated data for EduBench, please fill out the table below and send it to directionai@163.com  
[Human Annotated Data](https://1drv.ms/x/c/b1bbd011a3bb6457/EQzDBJxEPHFBmq91RoejDr8BnJ7qw7Ffbs27qJtirSYleQ?e=GirULe)


## 🫣Citation
If you find our benchmark or evaluation pipeline useful or interesting, please cite our paper.

```
@misc{xu2025edubenchcomprehensivebenchmarkingdataset,
      title={EduBench: A Comprehensive Benchmarking Dataset for Evaluating Large Language Models in Diverse Educational Scenarios}, 
      author={Bin Xu and Yu Bai and Huashan Sun and Yiguan Lin and Siming Liu and Xinyue Liang and Yaolin Li and Yang Gao and Heyan Huang},
      year={2025},
      eprint={2505.16160},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.16160}, 
}
```