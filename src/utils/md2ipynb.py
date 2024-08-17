import json
import re
import nbformat as nbf

def text_to_notebook(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 创建一个新的notebook
    nb = nbf.v4.new_notebook()

    # 分割内容为单元格
    cells = re.split(r'\n```python\n|\n```\n', content)

    for i, cell_content in enumerate(cells):
        if i % 2 == 0:  # Markdown 单元格
            nb['cells'].append(nbf.v4.new_markdown_cell(cell_content.strip()))
        else:  # 代码单元格
            nb['cells'].append(nbf.v4.new_code_cell(cell_content.strip()))

    # 写入notebook文件
    with open(output_file, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

# 使用示例

input_files = ["02_NLP_Fundamentals.ipynb",        "03_Basic_knowledge_and_architectural_characteristics_of_LLM.ipynb",
            "04_LLM_Development_Fundamentals.ipynb",
            "05_Introduction_and_Setup_of_the_Experimental_Environment.ipynb",
            "06_Model_Inference_and_Function_calling.ipynb",
            "07_Prompt_engineering_ChatGPT_Prompt_Engineering.ipynb",
            "08_Model_Quantization_Techniques.ipynb"
            ]

root_path  = "/Users/xingqiangchen/TASK/llm-course-chen/notebooks/"
for name in input_files:
    input_file = root_path + name  # 您的输入文件名
    output_file = root_path + name  # 输出的Jupyter Notebook文件名

    text_to_notebook(input_file, output_file)
    print(f"Notebook has been created: {output_file}")