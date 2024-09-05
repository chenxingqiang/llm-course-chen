import textwrap
import os


def sanitize_filename(filename):
    return "".join(
        c if c.isalnum() or c in (" ", "_", "-") else "_" for c in filename
    ).rstrip()


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

    output_dir = "llm_course_timelines"
    os.makedirs(output_dir, exist_ok=True)

    for index, (lesson, course) in enumerate(courses, start=1):
        mermaid_code = generate_mermaid_gantt(courses, index)

        filename = f"{index:02d}_{sanitize_filename(course)}.md"

        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(f"# LLM Course Timeline: {course}\n\n")
            f.write(mermaid_code)

        print(
            f"Generated Mermaid code for course {index}: {course} in file: {filename}"
        )


if __name__ == "__main__":
    main()
