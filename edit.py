from PIL import Image
import os 
import os.path as osp 
import torch 
import diffusers
from agent_tool_aux import main_aux
from agent_tool_edit import main_edit
from agent_tool_generate import main_generate 
import gc 
from dotenv import load_dotenv
import json 
import requests
import base64
from photo_critique_action.client import get_azure_client 

load_dotenv()
api_key = os.getenv("AZURE_OPENAI_KEY")
url = "https://api.openai.com/v1/chat/completions"

def encode_image(image_path):
    if not os.path.exists(image_path):
        return None 
    with open(image_path,"rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def download_and_save_image(url,save_path):
    try:
        response = requests.get(url,stream = True)
        response.raise_for_status()
        with open(save_path,"wb") as f:
            for chunk in response.iter_content(chunk_size = 8192):
                f.write(chunk)
        
        im = cv2.imread(save_path)
        if im is not None:
            im = cv2.resize(im,(512,512))
            cv2.imwrite(save_path,im)
            return True 
        else:
            return False 
    except Exception as e:
        print(f"Error downloading and saving image: {e}")
        return False 
    except Exception as e:
        print(f"Error downloading and saving image: {e}")
        return False 

def command_parse(commands, text, text_bg, dir='inputs'):
    args = []
    generation_arg = None
    for i in range(len(commands)):
        command = commands[i]
        if command['tool'] == 'edit':
            k = len(args)
            if 'box' in command:
                if 'intbox' in command:
                    bb = command['box']
                    command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"], 'box': command['box']} }
            else:
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"]} }
            args.append(arg)

            arg = {"tool": "object_addition_anydoor",  "text": text, "text_bg": text_bg, 
                    "output": osp.join(dir, str(k+2)+'.png'), "output_mask": osp.join(dir, str(k+2)+'_mask.png'), 
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": command["edit"], "layout": osp.join(dir, str(k+1)+'_mask.png') } }
            args.append(arg)

            arg = {"tool": "replace_anydoor", "output": osp.join(dir, str(k+3)+'.png'), 
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": osp.join(dir, str(k+2)+'.png'),
                             "object_mask": osp.join(dir, str(k+2)+'_mask.png'), "mask": osp.join(dir, str(k+1)+'_mask.png'),  } }
            args.append(arg)
        elif command['tool'] == 'move':
            if 'intbox' in command:
                bb = command['box']
                command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
            k = len(args)
            arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"]} }
            args.append(arg)

            arg = {"tool": "remove", "output":  osp.join(dir, str(k+2)+'.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "mask": osp.join(dir, str(k+1)+'_mask.png')} }
            args.append(arg)

            arg = {"tool": "addition_anydoor", "output": osp.join(dir, str(k+3)+'.png'), 
                "input": {"image": osp.join(dir, str(k+2)+'.png'), "object": osp.join(dir, str(k)+'.png'), 
                        "object_mask": osp.join(dir, str(k+1)+'_mask.png'), "mask": command["box"]  } }
            args.append(arg)
        elif command['tool'] == 'addition':
            k = len(args)
            if 'intbox' in command:
                bb = command['box']
                command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
            arg = {"tool": "object_addition_anydoor",  "text": text, "text_bg": text_bg, 
                    "output": osp.join(dir, str(k+1)+'.png'), "output_mask": osp.join(dir, str(k+1)+'_mask.png'), 
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": command["input"], "layout": command["box"] } }
            args.append(arg)

            arg = {"tool": "addition_anydoor", "output": osp.join(dir, str(k+2)+'.png'), 
                "input": {"image": osp.join(dir, str(k)+'.png'), "object": osp.join(dir, str(k+1)+'.png'), 
                        "object_mask": osp.join(dir, str(k+1)+'_mask.png'), "mask": command["box"]  } }
            args.append(arg)
        elif command['tool'] == 'remove':
            k = len(args)
            arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"], "mask_threshold": command.get("mask_thr", 0.0)} }
            if 'box' in command:
                if 'intbox' in command:
                    bb = command['box']
                    command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
                arg['input']['box'] = command['box']
            args.append(arg)

            arg = {"tool": "remove", "output":  osp.join(dir, str(k+2)+'.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "mask": osp.join(dir, str(k+1)+'_mask.png')} }
            args.append(arg)
        elif command['tool'] == 'instruction':
            k = len(args)
            arg = {"tool": "instruction", "output": osp.join(dir, str(k+1)+'.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["text"]} }
            args.append(arg)
        elif command['tool'] == 'edit_attribute':
            k = len(args)
            if 'box' in command:
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"], 'box': command['box']} }
            else:
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"]} }
            args.append(arg)

            arg = {"tool": "attribute_diffedit", 
                    "output": osp.join(dir, str(k+2)+'.png'),  
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": command["input"], "object_mask": osp.join(dir, str(k+1)+'_mask.png'), "attr": command["text"] } }
            args.append(arg)
        
        elif command['tool'] in ['text_to_image_SDXL', 'image_to_image_SD2', 'layout_to_image_LMD', 'layout_to_image_BoxDiff', 'superresolution_SDXL']:
            generation_arg = command
            generation_arg["output"] = osp.join(dir, '0.png')
    
    k = len(args)
    arg = {"tool": "superresolution_SDXL", "input": {"image": osp.join(dir, str(k)+'.png')}, "output": osp.join(dir, str(k+1)+'.png')}
    args.append(arg)
    if generation_arg is not None:
        args = [generation_arg] + args
    return args


def main():
    data_list = []
    for i in range(1,7):
        try:
            with open(f'/home/daclai/GenArtist/photo-critique-action/reviewer/reviewer_{i}.json','r',encoding='utf-8') as f:
                data = json.load(f)
                data_list.extend(data)
        except FileNotFoundError:
            continue 
    try:
        with open(f'prompt/corrections.txt','r',encoding='utf-8') as f:
            correction_prompt = f.read()
    except:
        print("Error reading correction prompt")
        exit()
    
    for idx,item in enumerate(data_list):
        reviewer_id = item.get("Reviewer ID")
        post_id = item.get("Post_ID")
        image_url = item.get("Image_URL")
        critique = item.get("Critique")
        status = item.get("Status")
        if status != "Submitted":
            print(f"Skip {post_id}: Status is '{status}' (not 'Submitted')")
            continue 
        reviewer_output_dir = osp.join("output",f'reviewer_{reviewer_id}')
        os.makedirs(reviewer_output_dir,exist_ok=True)
        final_output_path = osp.join(reviewer_output_dir,f'{post_id}.png')
        image_path = 'inputs/0.png'
        if not download_and_save_image(image_url,image_path):
            print(f"Error downloading and saving image: {image_url}")
            continue
        base64_image = encode_image(image_path)
        if not base64_image:
            print(f"Skip {post_id}: Error encoding image.")
            continue
        user_textprompt = (
            f"Critique for editing: {critique}\n"
            "Please analyze the critique and image and output only the editing operations as a Python list of dicts. "
            "Return the result with only plain text, do not use any markdown or other style. "
            "All characters must be in English. The available tools are in the following format:\n"
            f"{correction_prompt}" 
        )
        payload = json.dumps({
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {'type': 'text', 'text': user_textprompt}, 
                        {'type': 'image_url', 'image_url': {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        })

        try:
            response = response.request("POST")
        


        


    

