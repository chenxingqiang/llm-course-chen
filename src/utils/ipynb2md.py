import json
import nbformat as nbf
import os


def notebook_to_markdown(input_file, output_file):
    # 读取notebook文件
    with open(input_file, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    # 创建一个列表来存储Markdown内容
    md_content = []

    # 遍历notebook的每个单元格
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            # 对于Markdown单元格,直接添加内容
            md_content.append(cell.source)
        elif cell.cell_type == "code":
            # 对于代码单元格,添加代码块格式
            md_content.append("```python")
            md_content.append(cell.source)
            md_content.append("```")

        # 在每个单元格之后添加一个空行
        md_content.append("")

    # 将内容写入Markdown文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))


# 使用示例
input_files = [
    "01_Course_Overview.ipynb",
    "02_NLP_Fundamentals.ipynb",
    "03_Basic_knowledge_and_architectural_characteristics_of_LLM.ipynb",
    "04_LLM_Development_Fundamentals.ipynb",
    "05_Introduction_and_Setup_of_the_Experimental_Environment.ipynb",
    "06_Model_Inference_and_Function_calling.ipynb",
    "07_Prompt_engineering_ChatGPT_Prompt_Engineering.ipynb",
    "08_Model_Quantization_Techniques.ipynb",
]

root_path = "/Users/xingqiangchen/TASK/llm-course-chen/notebooks/"

for name in input_files:
    input_file = os.path.join(root_path, name)
    output_file = os.path.join(root_path, name.replace(".ipynb", ".md"))

    notebook_to_markdown(input_file, output_file)
    print(f"Markdown file has been created: {output_file}")
