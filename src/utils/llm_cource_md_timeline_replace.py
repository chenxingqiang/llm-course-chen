import os
import re

from generate_mermaid_gantt import generate_mermaid_gantt


def update_markdown_file(file_path, new_mermaid_code):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # 查找第一个 "# 1. xxx" 格式的标题
    title_pattern = r"^# 1\. .*$"
    title_match = re.search(title_pattern, content, re.MULTILINE)

    if title_match:
        title_end = title_match.end()
        before_title = content[:title_end]
        after_title = content[title_end:]

        # 检查标题后是否已存在 Mermaid 代码
        existing_mermaid = re.search(r"```mermaid.*?```", after_title, re.DOTALL)

        if existing_mermaid:
            # 替换现有的 Mermaid 代码
            updated_content = (
                before_title
                + "\n\n"
                + new_mermaid_code
                + after_title[: existing_mermaid.start()]
                + after_title[existing_mermaid.end() :]
            )
        else:
            # 在标题后插入新的 Mermaid 代码
            updated_content = before_title + "\n\n" + new_mermaid_code + after_title
    else:
        # 如果没有找到标题，则在文件开头插入 Mermaid 代码
        updated_content = new_mermaid_code + "\n\n" + content

    # 安全写入: 首先写入临时文件，然后替换原文件
    temp_file_path = file_path + ".temp"
    with open(temp_file_path, "w", encoding="utf-8") as temp_file:
        temp_file.write(updated_content)

    # 替换原文件
    os.replace(temp_file_path, file_path)


def main():
    courses = [
        ("lesson 1", "Course Overview"),
        ("lesson 2", "NLP Fundamentals"),
        ("lesson 3", "Basic knowledge and architectural characteristics of LLM"),
        ("lesson 4", "LLM Development Fundamentals"),
        ("lesson 5", "Introduction and Setup of the Experimental Environment"),
        ("lesson 6", "The concept of the tokenizer and common types"),
        ("lesson 7", "Text data preprocessing and preparation"),
        ("lesson 8", "LLM training - Fine-tuning"),
        ("lesson 9", "LLM training - Reward Modeling and Proximal Policy Optimization"),
        ("lesson 10", "Famous SOTA LLM models and JAIS model"),
        ("lesson 11", "Methods and Metrics for Model Evaluation"),
        ("lesson 12", "Model Inference and Function calling"),
        ("lesson 13", "Prompt engineering - ChatGPT Prompt Engineering"),
        ("lesson 14", "Model Quantization Techniques"),
        ("lesson 15", "Introduction to Chatbot Project"),
        ("lesson 16", "Test Dataset Collection and Model Evaluation"),
        ("lesson 17", "Designing input and output formats for chatbot with context"),
        ("lesson 18", "Model Deployment and Backend Development"),
        ("lesson 19", "Frontend web page debugging"),
        ("lesson 20", "System Testing and Deployment"),
        ("lesson 21", "RAG Introduction"),
        (
            "lesson 22",
            "RAG Frameworks - Introduction and use of Llamaindex and LangChain",
        ),
        ("lesson 23", "RAG embedding model"),
        ("lesson 24", "VectorDB - The use of the Milvus database"),
        ("lesson 25", "Keyword search and Vector Retrieval"),
        ("lesson 26", "Overview of the RAG project and RAG framework"),
        ("lesson 27", "Data preparation and preprocessing"),
        ("lesson 28", "Build the vector database with Milvus for Data"),
        ("lesson 29", "Load the model or Use OpenAI API"),
        ("lesson 30", "Implement RAG proxy service"),
        ("lesson 31", "Model testing and performance tuning"),
        ("lesson 32", "Implement frontend and the backend and their interface"),
        ("lesson 33", "Compose test cases for all public APIs"),
        (
            "lesson 34",
            "Package the project as a Docker image / Conda environment and deploy it in the cloud",
        ),
    ]

    md_dir = "/Users/xingqiangchen/TASK/llm-course-chen/notebooks/md"

    for index, (lesson, course) in enumerate(courses, start=1):
        mermaid_code = generate_mermaid_gantt(courses, index)

        # 查找匹配的文件
        lesson_prefix = f"lesson_{index:02d}"
        matching_files = [
            f
            for f in os.listdir(md_dir)
            if f.startswith(lesson_prefix) and f.endswith(".md")
        ]

        if matching_files:
            file_path = os.path.join(md_dir, matching_files[0])
            update_markdown_file(file_path, mermaid_code)
            print(f"Updated Mermaid code in file: {matching_files[0]}")
        else:
            print(f"No matching file found for {lesson_prefix}")


if __name__ == "__main__":
    main()
