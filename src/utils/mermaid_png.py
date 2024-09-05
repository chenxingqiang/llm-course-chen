import datetime
import os
import re
import nbformat as nbf
import base64
import requests


def mermaid_to_image_url(mermaid_code):
    graphbiz = mermaid_code
    request = {"code": graphbiz, "mermaid": {"theme": "default"}, "outputFormat": "svg"}
    response = requests.post("https://kroki.io/mermaid/svg", json=request)
    if response.status_code == 200:
        return f"data:image/svg+xml;base64,{base64.b64encode(response.content).decode('utf-8')}"
    else:
        return None


def text_to_notebook(input_file, output_file, lesson_num):
    # 读取输入文件
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 创建一个新的notebook
    nb = nbf.v4.new_notebook()

    # 定义正则表达式来匹配代码块和非代码块部分
    pattern = re.compile(r"```(.*?)\n(.*?)```", re.DOTALL)
    last_pos = 0
    mermaid_count = 0  # 计数mermaid图表数量

    for match in pattern.finditer(content):
        # 处理代码块之前的Markdown部分
        markdown_text = content[last_pos : match.start()].strip()
        if markdown_text:
            nb["cells"].append(nbf.v4.new_markdown_cell(markdown_text))

        # 处理代码块部分
        code_type = match.group(1).strip()  # 获取代码类型（例如 'python', 'mermaid'）
        code_text = match.group(2).strip()

        if code_type.lower() == "python":
            nb["cells"].append(nbf.v4.new_code_cell(code_text))
        elif code_type.lower() == "mermaid":
            mermaid_count += 1
            image_url = mermaid_to_image_url(code_text)
            if image_url:
                img_markdown = f"![Mermaid diagram {mermaid_count}]({image_url})"
                nb["cells"].append(nbf.v4.new_markdown_cell(img_markdown))
            else:
                # 如果图片生成失败，将 Mermaid 代码作为普通的代码块插入
                nb["cells"].append(
                    nbf.v4.new_markdown_cell(f"```mermaid\n{code_text}\n```")
                )
        else:
            # 如果不是Python代码或Mermaid代码块，可以将其作为普通的Markdown代码块
            nb["cells"].append(
                nbf.v4.new_markdown_cell(f"```{code_type}\n{code_text}\n```")
            )

        # 更新last_pos到当前匹配位置的末尾
        last_pos = match.end()

    # 处理文件结尾剩余的Markdown部分
    remaining_markdown = content[last_pos:].strip()
    if remaining_markdown:
        nb["cells"].append(nbf.v4.new_markdown_cell(remaining_markdown))

    # 写入notebook文件
    with open(output_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)


# 主程序
input_files = [
    "lesson_00_Course_Overview.md",
    "lesson_01_Course_Overview2.md",
    "lesson_02_NLP_Fundamentals.md",
    "lesson_03_Basic_knowledge_and_architectural_characteristics_of_LLM.md",
    "lesson_04_LLM_Development_Fundamentals.md",
    "lesson_05_Introduction_and_Setup_of_the_Experimental_Environment.md",
    "lesson_12_Model_Inference_and_Function_calling.md",
    "lesson_13_Prompt_engineering_ChatGPT_Prompt_Engineering.md",
    "lesson_14_Model_Quantization_Techniques.md",
    "lesson_17_Designing_Input_and_Output_Formats_for_Chatbot_with_Context.md",
    "lesson_18_Model_Deployment_and_Backend_Development.md",
    "lesson_19_Frontend_Web_Page_Debugging.md",
    "lesson_20_System_Testing_and_Deployment.md",
]

input_path = "/Users/xingqiangchen/TASK/llm-course-chen/notebooks/md/"
# 设置路径
output_base_path = "/Users/xingqiangchen/TASK/llm-course-chen/notebooks/"
timestamp = datetime.datetime.now().strftime(
    "ipynb-png-%Y%m%d%H%M"
)  # 生成时间戳，如 ipynb-2024082205
output_path = os.path.join(output_base_path, timestamp)

# 如果不存在该路径，则创建它
if not os.path.exists(output_path):
    os.makedirs(output_path)

for name in input_files:
    # 创建特定课程文件夹
    lesson_num = name.split("_")[1]
    lesson_folder = os.path.join(output_path, f"lesson_{lesson_num}")

    if not os.path.exists(lesson_folder):
        os.makedirs(lesson_folder)

    # 拼接输入文件的路径
    input_file = os.path.join(input_path, name)

    # 将输出文件的扩展名改为 .ipynb 并保存在课程子文件夹中
    output_file = os.path.join(lesson_folder, name.replace(".md", ".ipynb"))

    # 将文本转换为笔记本并生成Mermaid图表
    text_to_notebook(input_file, output_file, lesson_num)
    print(f"Notebook has been created: {output_file}")
