import textwrap


def generate_mermaid_gantt(courses, current_index, window_size=10):
    start_index = max(
        0, min(current_index - window_size // 2, len(courses) - window_size)
    )
    end_index = start_index + window_size

    mermaid_code = "```mermaid\ngantt\n    title LLM Course Timeline\n    dateFormat X\n    axisFormat %d\n    section Course Content\n"

    for i, (lesson, course) in enumerate(
        courses[start_index:end_index], start=start_index + 1
    ):
        course_id = f"a{i}"
        is_active = "active," if i == current_index else ""
        after = f"after a{i-1}" if i > start_index + 1 else "0"
        mermaid_code += f"    {course:<50} :{is_active}{course_id}, {after}, 1d\n"

    mermaid_code += "    section Lessons\n"

    for i in range(start_index + 1, end_index + 1):
        lesson_id = f"l{i}"
        is_active = "active," if i == current_index else ""
        after = f"after l{i-1}" if i > start_index + 1 else "0"
        mermaid_code += f"    lesson {i:<2} :{is_active}{lesson_id}, {after}, 1d\n"

    mermaid_code += "```\n"
    return mermaid_code


def generate_mermaid_gantt2(courses, current_index, window_size=10):
    start_index = max(
        0, min(current_index - window_size // 2, len(courses) - window_size)
    )
    end_index = start_index + window_size

    mermaid_code = textwrap.dedent(
        """
    ```mermaid
    gantt
        title LLM Course Timeline
        dateFormat X
        axisFormat %d
        section Course Content
    """
    )

    for i, (lesson, course) in enumerate(
        courses[start_index:end_index], start=start_index + 1
    ):
        course_id = f"a{i}"
        is_active = "active," if i == current_index else ""
        after = f"after a{i-1}" if i > start_index + 1 else "0"
        mermaid_code += f"    {course:<50} :{is_active}{course_id}, {after}, 1d\n"

    mermaid_code += "    section Lessons\n"

    for i in range(start_index + 1, end_index + 1):
        lesson_id = f"l{i}"
        is_active = "active," if i == current_index else ""
        after = f"after l{i-1}" if i > start_index + 1 else "0"
        mermaid_code += f"    lesson {i:<2} :{is_active}{lesson_id}, {after}, 1d\n"

    mermaid_code += "```"
    return mermaid_code