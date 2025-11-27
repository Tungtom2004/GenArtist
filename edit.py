import os
import json
import base64
import requests
import cv2
from agent_tool_aux import main_aux
from agent_tool_edit import main_edit
from agent_tool_generate import main_generate
from agent_tool import command_parse
import time
import re
from dotenv import load_dotenv
from PIL import Image
import io 
load_dotenv()

# ============================
# CONFIG
# ============================

AZURE_OPENAI_KEY      = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_VERSION  = os.getenv("AZURE_OPENAI_VERSION", "2024-10-21")
AZURE_OPENAI_DEPLOY   = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_CACHE = "./hf_cache"

# Inject vào môi trường cho Diffusers
os.environ["HF_HOME"] = HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN or ""
os.environ["HF_TOKEN"] = HF_TOKEN or ""

# Cho phép tải các model yêu cầu license (như SDXL)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["DIFFUSERS_OFFLINE"] = "0"

# Kiểm tra
print("Using HF_CACHE:", HF_CACHE)
print("HF_TOKEN is set:", HF_TOKEN is not None)

INPUT_FOLDER  = "inputs"
OUTPUT_FOLDER = "tung_ga"

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)

# ===========================================================
# ENCODING IMAGE
# ===========================================================

def encode_image(image_path):
    """Return a compressed thumbnail JPEG as base64 string to keep payloads small."""
    if not os.path.exists(image_path):
        return None
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.thumbnail((256, 256))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=40)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"encode_image error: {e}")
        return None


# ===========================================================
# DOWNLOAD IMAGE
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


# ===========================================================
# EDITING PROMPT — 1 LINE, SAFEST FOR AZURE
# ===========================================================

EDITING_PROMPT = (
    "Your task is to convert the CRITIQUE into a JSON array of image-editing commands. "
    "Each command MUST be a valid JSON object. Never use Python syntax. Never use single quotes. "
    "Allowed tools: "
    "{\"tool\":\"instruction_editing_MagicBrush\",\"input\":{\"image\":\"input.png\",\"text\":\"<instruction>\"}}, "
    "{\"tool\":\"remove_lama\",\"input\":{\"image\":\"input.png\",\"object\":\"<object>\"}}, "
    "{\"tool\":\"addition_anydoor\",\"input\":{\"image\":\"input.png\",\"object\":\"<object>\",\"mask\":\"TBG\"}}, "
    "{\"tool\":\"replace_anydoor\",\"input\":{\"image\":\"input.png\",\"object\":\"<new object>\",\"mask\":\"TBG\"}}, "
    "{\"tool\":\"attribute_diffedit\",\"input\":{\"image\":\"input.png\",\"object\":\"<object>\",\"attr\":\"<new attribute>\"}}, "
    "{\"tool\":\"drag_dragondiffusion\",\"input\":{\"image\":\"input.png\",\"p1\":\"TBG\",\"p2\":\"TBG\"}}. "
    "RULES: Always return ONLY a JSON array. No comments. No natural language. No markdown."
)


# ===========================================================
# CALL AZURE GPT-4o
# ===========================================================

def get_edit_commands(critique, base64_img):
    # CLEAN TEXT TO PREVENT AZURE ERRORS
    clean_critique = critique.replace("\n", " ").replace("\r", " ").replace('"', "'").strip()
    clean_b64 = base64_img.strip()

    user_prompt = (
        EDITING_PROMPT +
        " CRITIQUE: " + clean_critique +
        " IMAGE_BASE64: " + clean_b64 +
        " RETURN ONLY A JSON ARRAY."
    )

    # Safety: avoid extremely large payloads
    if len(user_prompt) > 200_000:
        print(f"Error: composed prompt too large ({len(user_prompt)} chars). Abort to avoid 400.")
        return None

    payload = {
        "messages": [
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0
    }

    try:
        url = (
            f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}"
            f"/openai/deployments/{AZURE_OPENAI_DEPLOY}/chat/completions?api-version={AZURE_OPENAI_VERSION}"
        )
        headers = {
            "api-key": AZURE_OPENAI_KEY,
            "Content-Type": "application/json"
        }

        # Debug
        print("Calling Azure OpenAI URL:", url)
        print("Payload size (chars):", len(json.dumps(payload)))

        # Retry loop (exponential backoff), honor Retry-After if provided
        max_retries = 6
        backoff = 1
        r = None
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=60)
            except Exception as e:
                print(f"Request exception (attempt {attempt}): {e}")
                if attempt == max_retries:
                    print("Max retries reached for request exceptions.")
                    return None
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            if r.status_code == 200:
                break

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                try:
                    wait = int(retry_after)
                except Exception:
                    wait = backoff
                print(f"Azure returned 429 (attempt {attempt}/{max_retries}), retrying after {wait}s")
                time.sleep(wait)
                backoff = min(backoff * 2, 60)
                continue

            # other non-200 -> log body and abort
            print("Azure returned status:", r.status_code)
            print("Response body:", r.text)
            return None

        if r is None:
            print("No response after retries.")
            return None

        # Try to get assistant content (safe)
        try:
            raw = r.json().get("choices", [])[0].get("message", {}).get("content", "").strip()
        except Exception:
            raw = r.text

        # Extract JSON array if wrapped in fences or other text
        def extract_json_array(text):
            m = re.search(r"``[json\\s*(.*?)\\s*](http://_vscodecontentref_/3)``", text, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                return text[start:end+1]
            return text

        cleaned = extract_json_array(raw)

        try:
            commands = json.loads(cleaned)
        except Exception:
            print("GPT returned invalid JSON. Raw response:")
            print(raw[:2000])
            print("Cleaned attempt (first 2000 chars):")
            print(cleaned[:2000])
            return None

        if not isinstance(commands, list):
            print("GPT returned non-list structure:", type(commands), commands)
            return None

        return commands

    except Exception as e:
        print(f"Error calling Azure GPT-4o: {e}")
        return None


# ===========================================================
# PROCESS REVIEWER
# ===========================================================

# ===========================================================
# PROCESS REVIEWER (ĐÃ SỬA)
# ===========================================================

def process_reviewer():

    file_path = f"dataset.json"
    if not os.path.exists(file_path):
        print(f"Reviewer file not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # Tất cả ảnh sẽ lưu vào 1 folder duy nhất
    output_dir = OUTPUT_FOLDER

    for entry in data_list:

        post_id = entry.get("Post_ID")
        critique = entry.get("Critique")
        image_url = entry.get("Image_URL")
        print(f"\nProcessing {post_id} ...")
        start_time = time.time()

        # ==== tải ảnh
        input_path = f"{INPUT_FOLDER}/0.png"
        if not download_and_save_image(image_url, input_path):
            print(f"Skipping {post_id}: image download failed")
            continue

        b64_img = encode_image(input_path)
        if not b64_img:
            print(f"Skipping {post_id}: encode failed")
            continue

        # ==== GPT tạo command
        commands = get_edit_commands(critique, b64_img)
        if commands is None:
            print(f"Skipping {post_id}: GPT returned invalid commands")
            continue

        # ==== build agent sequence
        text = "A detailed photo"
        text_bg = "A detailed photo"
        seq_args = command_parse(commands, text, text_bg)

        # OUTPUT FILE (yêu cầu của bạn)
        output_path = f"{output_dir}/{post_id}_output_genartist.png"

        seq_args.append({
            "tool": "superresolution_SDXL",
            "input": {"image": f"{INPUT_FOLDER}/{len(seq_args)-1}.png"},
            "output": output_path
        })

        print(f"Running {len(seq_args)} steps...")

        # ==== chạy từng bước
        for step in seq_args:
            json.dump(step, open("input.json", "w"))
            tool = step["tool"]

            if tool in ["object_addition_anydoor", "segmentation", "detection"]:
                os.system("python agent_tool_aux.py --json_out True")
            elif tool in ["addition_anydoor", "replace_anydoor", "remove",
                          "instruction", "attribute_diffedit"]:
                os.system("python agent_tool_edit.py --json_out True")
            elif tool in ["text_to_image_SDXL", "image_to_image_SD2",
                          "layout_to_image_LMD", "layout_to_image_BoxDiff",
                          "superresolution_SDXL"]:
                os.system("python agent_tool_generate.py --json_out True")

        # tính thời gian
        end_time = time.time()
        elapsed = end_time - start_time

        print(f"Done: {output_path}")
        print(f"Edit time: {elapsed:.2f} seconds")



# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":
    for i in range(1, 7):
        process_reviewer(i)
