这是一个将Jupyter Notebook (.ipynb)文件转换为Markdown (.md)文件的Python脚本。这个脚本将使用`nbformat`库来读取notebook文件,然后将其内容转换为Markdown格式。以下是实现这个功能的代码:



```python
import json
import nbformat as nbf
import os

def notebook_to_markdown(input_file, output_file):
    # 读取notebook文件
    with open(input_file, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # 创建一个列表来存储Markdown内容
    md_content = []

    # 遍历notebook的每个单元格
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            # 对于Markdown单元格,直接添加内容
            md_content.append(cell.source)
        elif cell.cell_type == 'code':
            # 对于代码单元格,添加代码块格式
            md_content.append('```python')
            md_content.append(cell.source)
            md_content.append('```')
        
        # 在每个单元格之后添加一个空行
        md_content.append('')

    # 将内容写入Markdown文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))

# 使用示例
input_files = [
    "02_NLP_Fundamentals.ipynb",
    "03_Basic_knowledge_and_architectural_characteristics_of_LLM.ipynb",
    "04_LLM_Development_Fundamentals.ipynb",
    "05_Introduction_and_Setup_of_the_Experimental_Environment.ipynb",
    "06_Model_Inference_and_Function_calling.ipynb",
    "07_Prompt_engineering_ChatGPT_Prompt_Engineering.ipynb",
    "08_Model_Quantization_Techniques.ipynb"
]

root_path = "/Users/xingqiangchen/TASK/llm-course-chen/notebooks/"

for name in input_files:
    input_file = os.path.join(root_path, name)
    output_file = os.path.join(root_path, name.replace('.ipynb', '.md'))

    notebook_to_markdown(input_file, output_file)
    print(f"Markdown file has been created: {output_file}")

```

这个脚本实现了将Jupyter Notebook (.ipynb)文件转换为Markdown (.md)文件的功能。以下是对代码的简要解释:

1. 我们导入了必要的库:`json`用于处理JSON格式,`nbformat`用于读取notebook文件,`os`用于处理文件路径。

2. `notebook_to_markdown`函数接受输入的.ipynb文件和输出的.md文件路径作为参数。

3. 在函数内,我们首先读取notebook文件,然后遍历其中的每个单元格。

4. 对于Markdown单元格,我们直接添加其内容到输出列表。

5. 对于代码单元格,我们在内容前后添加Python代码块标记(```)。

6. 在每个单元格之后,我们添加一个空行以保持格式清晰。

7. 最后,我们将所有内容写入到指定的Markdown文件中。

8. 在主程序部分,我们遍历给定的输入文件列表,对每个文件执行转换操作,并将结果保存为同名的.md文件。

要使用这个脚本,您只需要确保已经安装了`nbformat`库(可以通过`pip install nbformat`安装),然后运行这个脚本。它将处理指定目录下的所有列出的.ipynb文件,并在同一目录下创建对应的.md文件。