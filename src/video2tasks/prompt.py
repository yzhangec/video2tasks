def prompt_video_overview(n_images: int) -> str:
    """Prompt for generating a high-level task description and scene description for the whole video."""
    return (
        f"You are a robotic vision analyst. You are given {n_images} frames uniformly sampled from an entire robot manipulation video.\n\n"
        "Your job: provide TWO descriptions that capture the full video at a high level.\n\n"
        "1. **task_description**: A concise description of the complete task the robot accomplishes from start to finish "
        "(e.g., 'Arrange various fruits from a basket onto a white plate').\n\n"
        "2. **scene_description**: A brief description of the physical workspace — what objects are present, the tabletop layout, "
        "and relevant environmental context as observed at the start of the video "
        "(e.g., 'A table with a white plate on the left, a wicker basket in the center containing red apples, yellow bananas, and green pears').\n\n"
        "### Key Rules\n"
        "- **task_description**: focus on the overall goal, not individual sub-steps.\n"
        "- **scene_description**: describe the **initial scene** (before the robot acts). Be specific about object colors, types, and positions.\n"
        "- **No robot visible**: If the robot arm is **not visible** in any of the frames, do NOT hallucinate a task. "
        "Set `task_description` to `\"No action\"` and `scene_description` to a plain description of what is visible.\n\n"
        "### Output Format: Strict JSON\n"
        "Respond with ONLY a JSON object with three fields:\n"
        '{"thought": "<brief observation>", "task_description": "<overall task>", "scene_description": "<workspace scene>"}\n\n'
        "Example:\n"
        '{"thought": "The video shows a robot arm picking up various fruits one by one from a wicker basket and placing them onto a white plate.", '
        '"task_description": "Arrange fruits from a basket onto a white plate", '
        '"scene_description": "A tabletop scene with a white plate on the left, a wicker basket in the center containing red apples, yellow bananas, and green pears."}'
    )


def prompt_label_segment(n_images: int) -> str:
    """Prompt for labeling a single task segment with exactly one instruction."""
    return (
        f"You are a robotic vision analyzer watching a {n_images}-frame video clip.\n"
        "The robot is performing exactly ONE atomic manipulation task throughout this entire clip.\n\n"
        "Your job: describe this single task as a short imperative instruction.\n\n"
        "### Object Description Rules\n"
        "- The **manipulated object** is the one **stably grasped** by the robot arm — not merely an object that appears near the arm due to visual overlap.\n"
        "- If the robot arm is **not visible** in the clip, output `\"No action\"` as the instruction. Do NOT invent a task.\n"
        "- Include **color** (e.g., 'red apple', 'yellow banana', 'green pear').\n"
        "- Include **category/type** (e.g., apple, banana, plate, basket).\n"
        "- Avoid vague terms like 'the object' or 'the item'.\n"
        "- Example good instructions: 'Place the red apple on the white plate', "
        "'Pick up the yellow banana from the basket'.\n\n"
        "### Output Format: Strict JSON\n"
        "Respond with ONLY a JSON object with two fields:\n"
        '{"thought": "<one sentence observation>", "instruction": "<action instruction>"}\n\n'
        "Example:\n"
        '{"thought": "The robot picks up the red apple and places it on the white plate.", '
        '"instruction": "Place the red apple on the white plate"}'
    )


def prompt_switch_detection(n_images: int) -> str:
    return (
        f"You are a robotic vision analyzer watching a {n_images}-frame video clip of household manipulation tasks.\n"
        f"**Mapping:** Image indices range from 0 to {n_images - 1}.\n\n"
        "### Goal\n"
        "Detect **Atomic Task Boundaries** (Switch Points).\n"
        # "A 'Switch' occurs strictly when the robot **completes** interaction with one object and **starts** interacting with a DIFFERENT object.\n\n"
        "A 'Switch' occurs when:\n"
        "1. The robot finishes interacting with the current object (its hand clearly releases the object and remains away from it for several frames), even if it does not immediately start a new object.\n"
        "2. Or when the robot switches from one object to another (releases object A and starts interacting with object B).\n\n"
        "### Core Logic (The 'Distinct Object' Rule)\n"
        "**Key Definition — 'Interacting with an Object':** The robot is considered to be interacting with an object only when the object is **stably grasped** by the arm. "
        "An object that merely appears near the gripper due to visual overlap is **NOT** being interacted with — do not treat it as the manipulated object.\n\n"
        "**No robot visible:** If the robot arm is **not visible** in any frame of this clip, output no transitions and set the single instruction to `\"No action\"`. "
        "Do NOT hallucinate a task description.\n\n"
        "1. **True Switch:** Robot releases Object A (e.g., a cup) and moves to grasp Object B (e.g., a spoon). -> MARK SWITCH.\n"
        "2. **True Switch:** Robot releases Object A (e.g., a bottle) and idle for several frames. -> MARK SWITCH.\n"
        "3. **False Switch (IMPORTANT):** If the robot is manipulating different parts of the **SAME** object (e.g., folding sleeves then folding the body of the same shirt), this is **NOT** a switch. Treat it as one continuous task.\n"
        "4. **Visual Similarity:** Be careful with objects of the same color. Only mark a switch if you clearly see the robot **physically separate** from the first item before touching the second.\n\n"
        "### Hand Specification and Object Details\n"
        "Always **distinguish the robot's left and right hands** in your reasoning and instructions when possible.\n"
        "- Use explicit wording such as \"with the left hand\" or \"with the right hand\" when describing actions.\n"
        "- If it is visually ambiguous which hand is used, state that it is unclear.\n\n"
        "Be **as specific as possible** about the manipulated object:\n"
        "- Include **color** (e.g., \"red apple\", \"yellow banana\", \"green pear\").\n"
        "- Include **category/type** (e.g., apple, banana, plate, basket, cup).\n"
        "- Avoid using vague descriptions like \"the object\" or \"the item\". Always be specific.\n"
        "- Avoid using numerical descriptions like \"the first object\" or \"the second object\". Always be specific.\n"
        "- When multiple similar objects exist, describe anything that helps to uniquely identify the object (e.g., position on the table, size).\n\n"
        "In your `instructions`, always refer to objects with this detailed description (e.g., \"Place the red apple on the plate\", not just \"Place the fruit\").\n\n"
        "### Output Format: Strict JSON\n"
        "Your response must be a valid JSON object including a 'thought' field for step-by-step analysis, 'transitions' for the switch indices, and 'instructions' for the task labels.\n"
        "The length of the 'instructions' list MUST ALWAYS equal len(transitions) + 1.\n"
        "For example, if 'transitions' is [], then 'instructions' must contain exactly 1 element; "
        "if 'transitions' is [6], then 'instructions' must contain exactly 2 elements, etc.\n\n"
        "### Representative Examples\n"
        "**Example 1: Table Setting (True Switch)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-5: Robot places a fork. Frame 6: Hand releases fork and moves to the spoon. Frame 7: Hand grasps spoon. Switch detected at 6.\",\n"
        "  \"transitions\": [6],\n"
        "  \"instructions\": [\"Place the fork\", \"Place the spoon\"]\n"
        "}\n\n"
        "**Example 2: Folding Laundry (False Switch - Same Object)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-10: Robot folds the left sleeve of the black shirt. Frames 11-20: Robot folds the body of the **same** black shirt. Although the grasp changed, the object remains the same. The action is continuous.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Fold the black shirt\"]\n"
        "}\n\n"
        "**Example 3: Cleaning (Continuous)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-15: Robot is wiping the counter. The motion is repetitive, but it is the same task. No switch.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Wipe the counter\"]\n"
        "}"
    )