import os
import re
import nbformat as nbf
import datetime
import base64


from selenium.webdriver.support import expected_conditions as EC

from mermaid_to_image import mermaid_to_image


def mermaid_to_base64_image(mermaid_code, lesson_num, diagram_count):
    try:
        # Generate a file name
        file_name = f"lesson_{lesson_num}_mermaid_{diagram_count}.svg"
        output_file = os.path.join(output_path, f"lesson_{lesson_num}", file_name)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Use mermaid_to_image function to generate SVG
        if mermaid_to_image(mermaid_code, output_file, format="svg"):
            # Read the SVG file content
            with open(output_file, "rb") as f:
                svg_content = f.read()

            # Convert SVG content to base64
            base64_image = base64.b64encode(svg_content).decode("utf-8")

            return f"data:image/svg+xml;base64,{base64_image}", file_name
        else:
            raise Exception("Mermaid image generation failed")
    except Exception as e:
        print(f"Error generating Mermaid diagram: {e}")
        return None, None


def text_to_notebook(input_file, output_file, lesson_num):
    # Read the input file
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Create a new notebook
    nb = nbf.v4.new_notebook()

    # Define regex pattern to match code blocks and non-code parts
    pattern = re.compile(r"```(.*?)\n(.*?)```", re.DOTALL)
    last_pos = 0
    mermaid_count = 0  # Count Mermaid diagrams

    for match in pattern.finditer(content):
        # Process Markdown part before the code block
        markdown_text = content[last_pos : match.start()].strip()
        if markdown_text:
            nb["cells"].append(nbf.v4.new_markdown_cell(markdown_text))

        # Process code block part
        code_type = match.group(1).strip()  # Get code type (e.g., 'python', 'mermaid')
        code_text = match.group(2).strip()

        if code_type.lower() == "python":
            nb["cells"].append(nbf.v4.new_code_cell(code_text))
        elif code_type.lower() == "mermaid":
            mermaid_count += 1
            base64_image, file_name = mermaid_to_base64_image(
                code_text, lesson_num, mermaid_count
            )

            if base64_image:
                img_ref = (
                    f'<img src="{base64_image}" alt="Mermaid diagram {mermaid_count}">'
                )
                nb["cells"].append(nbf.v4.new_markdown_cell(img_ref))
                print(f"Mermaid diagram saved as: {file_name}")
            else:
                # If image generation fails, insert Mermaid code as a regular code block
                nb["cells"].append(
                    nbf.v4.new_markdown_cell(f"```mermaid\n{code_text}\n```")
                )
                print(
                    f"Failed to generate Mermaid diagram {mermaid_count}, inserted as code block"
                )
        else:
            # For non-Python and non-Mermaid code blocks, insert as regular Markdown code blocks
            nb["cells"].append(
                nbf.v4.new_markdown_cell(f"```{code_type}\n{code_text}\n```")
            )

        # Update last_pos to the end of the current match
        last_pos = match.end()

    # Process remaining Markdown at the end of the file
    remaining_markdown = content[last_pos:].strip()
    if remaining_markdown:
        nb["cells"].append(nbf.v4.new_markdown_cell(remaining_markdown))

    # Write the notebook file
    with open(output_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)


# Main program
# input_files = [
#     "lesson_00_Course_Overview.md",
#     "lesson_01_Course_Overview2.md",
#     "lesson_02_NLP_Fundamentals.md",
#     "lesson_03_Basic_knowledge_and_architectural_characteristics_of_LLM.md",
#     "lesson_04_LLM_Development_Fundamentals.md",
#     "lesson_05_Introduction_and_Setup_of_the_Experimental_Environment.md",
#     "lesson_12_Model_Inference_and_Function_calling.md",
#     "lesson_13_Prompt_engineering_ChatGPT_Prompt_Engineering.md",
#     "lesson_14_Model_Quantization_Techniques.md",
#     "lesson_17_Designing_Input_and_Output_Formats_for_Chatbot_with_Context.md",
#     "lesson_18_Model_Deployment_and_Backend_Development.md",
#     "lesson_19_Frontend_Web_Page_Debugging.md",
#     "lesson_20_System_Testing_and_Deployment.md",
# ]

input_files = [
        "lesson_1.md",
        "lesson_2.md",
        "lesson_3.md",
        "lesson_4.md",
        "lesson_5.md",
        "lesson_12.md",
        "lesson_13.md",
        "lesson_14.md",
        "lesson_17.md",
        "lesson_18.md",
        "lesson_19.md",
        "lesson_20.md",
    ]

input_path = "/Users/xingqiangchen/TASK/llm-course-chen/notebooks/md/"
output_base_path = "/Users/xingqiangchen/TASK/llm-course-chen/notebooks/"
timestamp = datetime.datetime.now().strftime("ipynb-svg-%Y%m%d%H%M")
output_path = os.path.join(output_base_path, timestamp)

# Create the output path if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

for name in input_files:
    # Create specific lesson folder
    lesson_num = name.split("_")[1]
    lesson_folder = os.path.join(output_path, f"lesson_{lesson_num}")
    if not os.path.exists(lesson_folder):
        os.makedirs(lesson_folder)

    # Construct input file path
    input_file = os.path.join(input_path, name)
    # Change output file extension to .ipynb and save in the lesson subfolder
    output_file = os.path.join(lesson_folder, name.replace(".md", ".ipynb"))

    # Convert text to notebook and generate Mermaid diagrams
    text_to_notebook(input_file, output_file, lesson_num)
    print(f"Notebook has been created: {output_file}")
