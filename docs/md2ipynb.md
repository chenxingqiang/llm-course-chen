Python脚本，用于将文本内容转换为Jupyter Notebook格式。这个脚本将读取输入文件，解析内容，并创建一个对应的.ipynb文件。以下是实现这个功能的代码：

```python
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
input_file = 'input.txt'  # 您的输入文件名
output_file = 'output.ipynb'  # 输出的Jupyter Notebook文件名

text_to_notebook(input_file, output_file)
print(f"Notebook has been created: {output_file}")
```

使用说明：

1. 首先，您需要安装 `nbformat` 库。您可以使用以下命令安装：
   ```
   pip install nbformat
   ```

2. 将您提供的课程内容保存为一个文本文件，例如 `input.txt`。

3. 在文本文件中，使用 ```python 和 ``` 来标记代码块的开始和结束。这些标记应该单独占一行。

4. 运行这个脚本。它会读取 `input.txt` 文件，并创建一个名为 `output.ipynb` 的Jupyter Notebook文件。

5. 您可以使用Jupyter Notebook或JupyterLab打开生成的 `output.ipynb` 文件。

这个脚本会将文本内容中的Markdown部分转换为Markdown单元格，将Python代码部分转换为代码单元格。它假设文本内容的结构是Markdown和代码块交替出现的。

如果您的输入文件结构有所不同，或者您需要更复杂的解析逻辑，可能需要对脚本进行相应的调整。