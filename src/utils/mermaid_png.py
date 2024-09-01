import os
import re
import nbformat as nbf
import datetime
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def save_mermaid_as_image(mermaid_code, output_file, method='auto'):
    # HTML template with Mermaid
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mermaid Diagram</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; }}
            .mermaid {{ display: flex; justify-content: center; align-items: center; min-height: 100vh; }}
        </style>
    </head>
    <body>
        <div class="mermaid">
        {mermaid_code}
        </div>
        <script>
            mermaid.initialize({{ startOnLoad: true }});
        </script>
    </body>
    </html>
    """

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Create a new Chrome driver
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Save the HTML to a temporary file
        with open("temp.html", "w") as f:
            f.write(html)

        # Load the HTML file
        driver.get("file://" + os.path.abspath("temp.html"))

        # Wait for the Mermaid diagram to render
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "mermaid"))
        )

        # Get the size of the Mermaid diagram and the window
        size = driver.execute_script("""
            var svg = document.querySelector('.mermaid svg');
            var body = document.body;
            return {
                svg: {
                    width: svg.getBBox().width,
                    height: svg.getBBox().height
                },
                window: {
                    width: body.scrollWidth,
                    height: body.scrollHeight
                }
            };
        """)

        # Decide whether to crop or keep full screenshot
        if method == 'auto':
            if size['svg']['width'] < size['window']['width'] * 0.9 and size['svg']['height'] < size['window']['height'] * 0.9:
                method = 'crop'
            else:
                method = 'full'

        if method == 'crop':
            # Set the viewport size to match the diagram size with a small margin
            margin = 10
            driver.set_window_size(size['svg']['width'] + margin*2, size['svg']['height'] + margin*2)
            
            # Scroll to ensure the diagram is in view
            driver.execute_script("window.scrollTo(0, 0);")
            
            # Get the Mermaid div
            mermaid_div = driver.find_element(By.CLASS_NAME, "mermaid")

            # Capture the screenshot of the Mermaid div
            mermaid_div.screenshot(output_file)
        else:  # full screenshot
            # Capture the full page screenshot
            driver.save_screenshot(output_file)

        print(f"Mermaid diagram saved as image: {output_file}")
        return True
    except Exception as e:
        print(f"Error generating PNG for Mermaid diagram: {e}")
        return False
    finally:
        driver.quit()
        os.remove("temp.html")
def text_to_notebook(input_file, output_file, lesson_num):
    # 读取输入文件
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 创建一个新的notebook
    nb = nbf.v4.new_notebook()

    # 获取输出文件所在的目录
    output_dir = os.path.dirname(output_file)

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
            png_file = os.path.join(
                output_dir, f"lesson_{lesson_num}_mermaid_{mermaid_count}.png"
            )

            if save_mermaid_as_image(code_text, png_file):
                img_ref = f"![Mermaid diagram](lesson_{lesson_num}_mermaid_{mermaid_count}.png)"
                nb["cells"].append(nbf.v4.new_markdown_cell(img_ref))
            else:
                # 如果图片生成失败，将 Mermaid 代码作为普通的代码块插入
                nb["cells"].append(nbf.v4.new_markdown_cell(f"```mermaid\n{code_text}\n```"))
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
    "lesson_01_Course_Overview.md",
    "lesson_02_NLP_Fundamentals.md",
    "lesson_03_Basic_knowledge_and_architectural_characteristics_of_LLM.md",
    "lesson_04_LLM_Development_Fundamentals.md",
    "lesson_05_Introduction_and_Setup_of_the_Experimental_Environment.md",
    "lesson_12_Model_Inference_and_Function_calling.md",
    "lesson_13_Prompt_engineering_ChatGPT_Prompt_Engineering.md",
    "lesson_14_Model_Quantization_Techniques.md",
]

input_path = "/Users/xingqiangchen/TASK/llm-course-chen/notebooks/md/"
# 设置路径
output_base_path = "/Users/xingqiangchen/TASK/llm-course-chen/notebooks/"
timestamp = datetime.datetime.now().strftime("ipynb-%Y%m%d%H%M")  # 生成时间戳，如 ipynb-2024082205
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