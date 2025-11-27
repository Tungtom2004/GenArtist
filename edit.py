import os
import json
import base64
import requests
import cv2
from agent_tool_aux import main_aux
from agent_tool_edit import main_edit
from agent_tool_generate import main_generate
from agent_tool import command_parse

# ============================
#  CONFIG (FILL THESE)
# ============================

AZURE_OPENAI_KEY      = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_VERSION  = os.getenv("AZURE_OPENAI_VERSION", "2024-10-21")
AZURE_OPENAI_DEPLOY   = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

INPUT_FOLDER  = "inputs"
OUTPUT_FOLDER = "outputs"

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===========================================================
# ENCODING IMAGE
# ===========================================================

def encode_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ===========================================================
# DOWNLOADING IMAGE
# ===========================================================

def download_and_save_image(url, save_path):
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        img = cv2.imread(save_path)
        if img is None:
            print("Error: cv2 cannot read image.")
            return False

        img = cv2.resize(img, (512, 512))
        cv2.imwrite(save_path, img)
        return True
    except Exception as e:
        print(f"Download image error: {e}")
        return False

EDITING_PROMPT = """
Your task is to convert the CRITIQUE into a sequence of image editing commands.
You MUST return a valid JSON array. 
Every element MUST be a JSON object. 
Do NOT return Python lists. 
Do NOT return single quotes. 
Do NOT return comments. 
Do NOT return explanations. 
Return ONLY a JSON array. 

-------------------------------------------------------------------------------
AVAILABLE TOOLS (choose the most appropriate):

1) instruction_editing_MagicBrush  
{"tool": "instruction_editing_MagicBrush",
 "input": {"image": "input.png", "text": "<instruction>"} }

2) remove_lama  
{"tool": "remove_lama",
 "input": {"image": "input.png", "object": "<object>"} }

3) addition_anydoor  
{"tool": "addition_anydoor",
 "input": {"image": "input.png", "object": "<object>", "mask": "TBG"} }

4) replace_anydoor  
{"tool": "replace_anydoor",
 "input": {"image": "input.png", "object": "<new object>", "mask": "TBG"} }

5) attribute_diffedit  
{"tool": "attribute_diffedit",
 "input": {"image": "input.png", "object": "<object>", "attr": "<new attribute>"} }

6) drag_dragondiffusion  
{"tool": "drag_dragondiffusion",
 "input": {"image": "input.png", "p1": "TBG", "p2": "TBG"} }

-------------------------------------------------------------------------------

RULES:
- ALWAYS return a JSON array.
- NEVER return Python.
- NEVER return text outside the JSON.
- NEVER use single quotes.
- If no edit is needed: return [].

-------------------------------------------------------------------------------
"""


def get_edit_commands(critique, base64_img):
    user_prompt = f"""
{EDITING_PROMPT}

CRITIQUE:
{critique}

IMAGE (base64):
{base64_img}

Return ONLY a JSON array.
""".strip()

    payload = {
        "model": AZURE_OPENAI_DEPLOY,
        "messages": [
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0
    }

    try:
        url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOY}/chat/completions?api-version={AZURE_OPENAI_VERSION}"
        headers = {
            "api-key": AZURE_OPENAI_KEY,
            "Content-Type": "application/json"
        }

        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()

        raw = r.json()["choices"][0]["message"]["content"].strip()

        # PARSE JSON
        try:
            commands = json.loads(raw)
        except:
            print("GPT returned invalid JSON. Raw:")
            print(raw)
            return None

        if not isinstance(commands, list):
            print("GPT returned non-list structure:", commands)
            return None

        return commands

    except Exception as e:
        print(f"Error calling Azure GPT-4o: {e}")
        return None


# ===========================================================
# PROCESS REVIEWER
# ===========================================================

def process_reviewer(reviewer_id):
    file_path = f"data/reviewer_{reviewer_id}.json"
    if not os.path.exists(file_path):
        print(f"Reviewer file not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    print(f"Processing reviewer {reviewer_id} with {len(data_list)} items")

    output_dir = f"{OUTPUT_FOLDER}/reviewer_{reviewer_id}"
    os.makedirs(output_dir, exist_ok=True)

    for entry in data_list:
        post_id = entry.get("Post_ID")
        critique = entry.get("Critique")
        image_url = entry.get("Image_URL")
        status = entry.get("Status", "").lower()

        if status == "pending":
            print(f"Skipping {post_id}: pending status")
            continue

        print(f"Processing {post_id}")

        input_path = f"{INPUT_FOLDER}/0.png"
        if not download_and_save_image(image_url, input_path):
            print(f"Skipping {post_id}: image download failed")
            continue

        b64_img = encode_image(input_path)
        if not b64_img:
            print(f"Skipping {post_id}: encode failed")
            continue

        commands = get_edit_commands(critique, b64_img)
        if commands is None:
            print(f"Skipping {post_id}: GPT returned invalid commands")
            continue

        # Build full sequence
        text = "A detailed photo"
        text_bg = "A detailed photo"
        seq_args = command_parse(commands, text, text_bg)

        # add a final SR
        seq_args.append({
            "tool": "superresolution_SDXL",
            "input": {"image": f"{INPUT_FOLDER}/{len(seq_args)-1}.png"},
            "output": f"{output_dir}/{post_id}_final.png"
        })

        print(f"Running {len(seq_args)} steps...")

        # EXECUTE
        for step in seq_args:
            json.dump(step, open("input.json", "w"))
            tool = step["tool"]

            if tool in ["object_addition_anydoor", "segmentation", "detection"]:
                os.system("python agent_tool_aux.py --json_out True")
            elif tool in ["addition_anydoor", "replace_anydoor", "remove", "instruction", "attribute_diffedit"]:
                os.system("python agent_tool_edit.py --json_out True")
            elif tool in ["text_to_image_SDXL", "image_to_image_SD2",
                          "layout_to_image_LMD", "layout_to_image_BoxDiff",
                          "superresolution_SDXL"]:
                os.system("python agent_tool_generate.py --json_out True")

        print(f"Done: {output_dir}/{post_id}_final.png")


# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":
    for i in range(1, 7):
        process_reviewer(i)
