import gradio as gr
import json
import pyautogui
import screeninfo
from transformers import Owlv2Processor, Owlv2ForObjectDetection, AutoModel, AutoTokenizer
from PIL import Image, ImageDraw
import torch
import os
import shutil
import re  # Import the re module for regular expression operations
import json
import time
import threading

oob_tasks_file = "extensions/Lucid_Autonomy/oob_tasks.json"

# Configurable variables
MONITOR_INDEX = 1  # Index of the monitor to capture (0 for the first monitor, 1 for the second monitor, etc.)
TEXT_QUERIES = "colored hyperlink text,clickable icons,text bar field,clickable UI buttons,UI tabs,Interactive UI Elements,blue text hyperlink"  # Comma-separated text queries for object detection
SCORE_THRESHOLD = 0.24  # Score threshold for filtering low-probability predictions
VISION_MODEL_QUESTION = "Keep your response to less than 5 sentences: What is the text in this image or what does this image describe? The AI is aiding in helping those with visual handicaps so the AI will focus on providing all of the relevant information for the listener."
FULL_IMAGE_VISION_MODEL_QUESTION = "This is a computer screenshot, identify all article text if the screen is displaying an article.  Describe what you see in this image, if you see UI elements describe them and their general location and if you see text read the text verbatim.  Describe the general theme and context of the image. You are aiding in helping those with visual handicaps so you need to try and focus on providing all of the relevant text on a screen and contextualizing it for the listener."

# Use GPU if available
device = 0

# Global variables for models
owlv2_model = None
owlv2_processor = None
minicpm_llama_model = None
minicpm_llama_tokenizer = None

VISION_MODEL_ID = "/home/myself/Desktop/miniCPM_llava3V/MiniCPM-V-2_6/"

def load_owlv2_model():
    global owlv2_model, owlv2_processor
    owlv2_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", trust_remote_code=True).to(device)
    owlv2_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble", trust_remote_code=True)
    print("Owlv2 model loaded.")

def unload_owlv2_model():
    global owlv2_model, owlv2_processor
    if owlv2_model is not None:
        del owlv2_model
        del owlv2_processor
        torch.cuda.empty_cache()
        print("Owlv2 model unloaded.")

def load_minicpm_llama_model():
    global minicpm_llama_model, minicpm_llama_tokenizer
    if "int4" in VISION_MODEL_ID:
        # Load the 4-bit quantized model and tokenizer
        minicpm_llama_model = AutoModel.from_pretrained(
            VISION_MODEL_ID,
            device_map={"": device},
            trust_remote_code=True,
            torch_dtype=torch.float16  # Use float16 as per the example code for 4-bit models
        ).eval()
        minicpm_llama_tokenizer = AutoTokenizer.from_pretrained(
            VISION_MODEL_ID,
            trust_remote_code=True
        )
        print("MiniCPM-Llama3 4-bit quantized model loaded on-demand.")
    else:
        # Load the standard model and tokenizer
        minicpm_llama_model = AutoModel.from_pretrained(
            VISION_MODEL_ID,
            device_map={"": device},  # Use the specified CUDA device
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).eval()
        minicpm_llama_tokenizer = AutoTokenizer.from_pretrained(
            VISION_MODEL_ID,
            trust_remote_code=True
        )
        print("MiniCPM-Llama3 standard model loaded on-demand.")

def unload_minicpm_llama_model():
    global minicpm_llama_model, minicpm_llama_tokenizer
    if minicpm_llama_model is not None:
        del minicpm_llama_model
        del minicpm_llama_tokenizer
        torch.cuda.empty_cache()
        print("MiniCPM-Llama3 model unloaded.")

def take_screenshot(monitor_index, text_queries, score_threshold, vision_model_question, full_image_vision_model_question, output_filename="screenshot.png"):
    """
    Takes a screenshot of a specific monitor in a multi-monitor setup.

    Args:
        monitor_index (int): The index of the monitor to capture.
        text_queries (str): Comma-separated text queries for object detection.
        score_threshold (float): The score threshold for filtering low-probability predictions.
        vision_model_question (str): The question to ask the vision model for cropped images.
        full_image_vision_model_question (str): The question to ask the vision model for the full image.
        output_filename (str): The filename to save the screenshot as (default is "screenshot.png").
    """
    monitor_index = int(monitor_index)
    score_threshold = float(score_threshold)

    # Get information about all monitors
    monitors = screeninfo.get_monitors()

    if monitor_index >= len(monitors):
        raise ValueError(f"Monitor index {monitor_index} is out of range. There are only {len(monitors)} monitors.")

    # Get the region of the specified monitor
    monitor = monitors[monitor_index]
    region = (monitor.x, monitor.y, monitor.width, monitor.height)

    # Take the screenshot
    output_dir = "extensions/Lucid_Autonomy"
    os.makedirs(output_dir, exist_ok=True)
    screenshot_path = os.path.join(output_dir, output_filename)
    screenshot = pyautogui.screenshot(region=region)
    screenshot.save(screenshot_path)
    print(f"Screenshot saved as {screenshot_path}")

    # Process the image using the Owlv2 model
    annotated_image_path, human_annotated_image_path, json_output_path = query_image(screenshot_path, text_queries, score_threshold)

    # Load the miniCPM-lama3-v-2_5 model
    load_minicpm_llama_model()

    # Process the entire screenshot with the minicpm model
    full_image_response = process_with_vision_model(annotated_image_path, full_image_vision_model_question)

    # Unload the miniCPM-lama3-v-2_5 model after processing the full image
    unload_minicpm_llama_model()

    # Create the output directory if it doesn't exist
    output_dir = "extensions/Lucid_Autonomy/ImageOutputTest"
    os.makedirs(output_dir, exist_ok=True)

    # Clear the output directory
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    # Crop the detected objects and save them to the output directory
    crop_images(annotated_image_path, json_output_path, output_dir, full_image_response)

    print("Processing complete. Check the output directory for the cropped images and results.")

def query_image(img_path, text_queries, score_threshold):
    """
    Processes the image using the Owlv2 model to detect objects based on text queries.

    Args:
        img_path (str): The path to the input image.
        text_queries (str): Comma-separated text queries for object detection.
        score_threshold (float): The score threshold for filtering low-probability predictions.
    """
    load_owlv2_model()

    img = Image.open(img_path)
    text_queries = text_queries.split(",")

    size = max(img.size)
    target_sizes = torch.Tensor([[size, size]])
    inputs = owlv2_processor(text=text_queries, images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = owlv2_model(**inputs)

    outputs.logits = outputs.logits.cpu()
    outputs.pred_boxes = outputs.pred_boxes.cpu()
    results = owlv2_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

    result_labels = []
    img_pil = img.copy()
    draw = ImageDraw.Draw(img_pil)

    human_img_pil = img.copy()
    human_draw = ImageDraw.Draw(human_img_pil)

    for box, score, label in zip(boxes, scores, labels):
        box = [int(i) for i in box.tolist()]
        if score < score_threshold:
            continue
        result_labels.append([box, text_queries[label.item()]])
        # Do not draw the text label within the bounding box
        # draw.text((box[0], box[1]), text_queries[label.item()], fill="red")

        # Draw the boxes for the human-annotated image
        human_draw.rectangle(box, outline="red", width=2)

    # Save the annotated image for debugging
    annotated_image_path = "extensions/Lucid_Autonomy/screenshot.png"
    img_pil.save(annotated_image_path)

    # Save the human-annotated image for human inspection
    human_annotated_image_path = "extensions/Lucid_Autonomy/Human_annotated_image.png"
    human_img_pil.save(human_annotated_image_path)

    # Save the JSON output to a file
    json_output_path = "extensions/Lucid_Autonomy/output.json"
    with open(json_output_path, 'w') as f:
        json.dump(result_labels, f)

    unload_owlv2_model()

    return annotated_image_path, human_annotated_image_path, json_output_path

def crop_images(image_path, json_output_path, output_dir="extensions/Lucid_Autonomy/ImageOutputTest", full_image_response=""):
    """
    Crops out the detected objects from the input image based on the bounding box coordinates.

    Args:
        image_path (str): The path to the input image.
        json_output_path (str): The path to the JSON file containing the bounding box coordinates.
        output_dir (str): The directory to save the cropped images (default is "extensions/Lucid_Autonomy/ImageOutputTest").
        full_image_response (str): The response from the vision model for the full image.
    """
    with open(json_output_path, 'r') as f:
        data = json.load(f)

    img = Image.open(image_path)

    results = []

    # Load the miniCPM-lama3-v-2_5 model once before processing the images
    load_minicpm_llama_model()

    for i, item in enumerate(data):
        box = item[0]
        label = item[1]
        cropped_img = img.crop(box)
        cropped_img_path = f"{output_dir}/{label}_{i}.png"
        cropped_img.save(cropped_img_path)

        # Process the cropped image with the miniCPM-lama3-v-2_5 model
        vision_model_response = process_with_vision_model(cropped_img_path, VISION_MODEL_QUESTION)

        # Store the results in a structured format
        results.append({
            "image_name": f"{label}_{i}.png",
            "coordinates": box,
            "description": vision_model_response,
            "children": []
        })

    # Unload the miniCPM-lama3-v-2_5 model after processing all images
    unload_minicpm_llama_model()

    # Determine parent-child relationships
    for i, result in enumerate(results):
        for j, other_result in enumerate(results):
            if i != j:
                box1 = result["coordinates"]
                box2 = other_result["coordinates"]
                center_x1 = (box1[0] + box1[2]) // 2
                center_y1 = (box1[1] + box1[3]) // 2
                if box2[0] <= center_x1 <= box2[2] and box2[1] <= center_y1 <= box2[3]:
                    other_result["children"].append(result["image_name"])

    # Save the results to a JSON file
    results_json_path = f"{output_dir}/results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Add the full image response to the results
    if full_image_response:
        results.append({
            "image_name": "full_image.png",
            "coordinates": [],
            "description": full_image_response,
            "children": []
        })

        # Save the updated results to the JSON file
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=4)

def process_with_vision_model(image_path, question):
    """
    Processes the image with the miniCPM-lama3-v-2_5 model and returns the response.

    Args:
        image_path (str): The path to the input image.
        question (str): The question to ask the vision model.

    Returns:
        str: The response from the vision model.
    """
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "user", "content": f"{question}"}
    ]
    # Define the generation arguments
    generation_args = {
        "max_new_tokens": 1024,
        "repetition_penalty": 1.05,
        "num_beams": 3,
        "top_p": 0.9,
        "top_k": 1,
        "temperature": 0.1,
        "sampling": True,
    }
    if "int4" in VISION_MODEL_ID:
        # Disable streaming for the 4-bit model
        generation_args["stream"] = False
        # Use the model.chat method with streaming enabled
        vision_model_response = ""
        for new_text in minicpm_llama_model.chat(
                image=image,
                msgs=messages,
                tokenizer=minicpm_llama_tokenizer,
                **generation_args
        ):
            vision_model_response += new_text
            print(new_text, flush=True, end='')

    else:
        minicpm_llama_model.to(device)
        vision_model_response = minicpm_llama_model.chat(
            image=image,
            msgs=messages,
            tokenizer=minicpm_llama_tokenizer,
            **generation_args
        )
    return vision_model_response

# Global variables for Gradio inputs
global_vars = {
    "global_monitor_index": MONITOR_INDEX,
    "global_text_queries": TEXT_QUERIES,
    "global_score_threshold": SCORE_THRESHOLD,
    "global_vision_model_question": VISION_MODEL_QUESTION,
    "global_full_image_vision_model_question": FULL_IMAGE_VISION_MODEL_QUESTION
}

def ui():
    """
    Creates custom gradio elements when the UI is launched.
    """
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                monitor_index = gr.Textbox(label="Monitor Index", value=str(MONITOR_INDEX))
                text_queries = gr.Textbox(label="Text Queries", value=TEXT_QUERIES)
                score_threshold = gr.Textbox(label="Score Threshold", value=str(SCORE_THRESHOLD))
                vision_model_question = gr.Textbox(label="Vision Model Question", value=VISION_MODEL_QUESTION)
                full_image_vision_model_question = gr.Textbox(label="Full Image Vision Model Question", value=FULL_IMAGE_VISION_MODEL_QUESTION)
                take_screenshot_button = gr.Button(value="Take Screenshot")

        # Update global variables when the user changes the input fields
        monitor_index.change(lambda x: global_vars.update({"global_monitor_index": int(x)}), inputs=monitor_index, outputs=None)
        text_queries.change(lambda x: global_vars.update({"global_text_queries": x}), inputs=text_queries, outputs=None)
        score_threshold.change(lambda x: global_vars.update({"global_score_threshold": float(x)}), inputs=score_threshold, outputs=None)
        vision_model_question.change(lambda x: global_vars.update({"global_vision_model_question": x}), inputs=vision_model_question, outputs=None)
        full_image_vision_model_question.change(lambda x: global_vars.update({"global_full_image_vision_model_question": x}), inputs=full_image_vision_model_question, outputs=None)

        take_screenshot_button.click(
            fn=take_screenshot,
            inputs=[monitor_index, text_queries, score_threshold, vision_model_question, full_image_vision_model_question],
            outputs=None
        )

    return demo

def input_modifier(user_input, state):
    """
    Modifies the user input before it is processed by the LLM.
    """
    return user_input

def output_modifier(output, state):
    """
    Modifies the LLM output before it is presented in the UI.
    """
    global oob_tasks

    # Search for the "Autonomy_Tasks:" trigger phrase in the LLM's output
    if "Autonomy_Tasks:" in output:
        # Clean the chat history
        state["history"] = history_modifier(state["history"])
        # Extract tasks and execute them
        tasks_section = output.split("Autonomy_Tasks:")[-1].strip()
        if tasks_section:
            # Write OOB tasks to a JSON file
            with open(oob_tasks_file, 'w') as f:
                json.dump(tasks_section.split("\n"), f)

            # Trigger the execution of OOB tasks after a delay
            threading.Thread(target=execute_oob_tasks_with_delay).start()

        # Append new compressed results
        results_json_path = "extensions/Lucid_Autonomy/ImageOutputTest/results.json"
        with open(results_json_path, 'r') as f:
            compressed_results = json.load(f)
        #output = append_compressed_results(output, compressed_results)

    return output

oob_tasks = []

def execute_oob_tasks(tasks_section):
    """
    Executes the OOB tasks listed in the tasks_section.
    """
    global oob_tasks
    tasks = tasks_section
    for task in tasks:
        task = task.strip()
        if task.startswith("OOB_MouseMover ="):
            image_name = task.split("=")[1].strip()
            move_mouse_to_image(image_name)
        elif task.startswith("OOB_MouseClick ="):
            value = task.split("=")[1].strip()
            if "," in value:  # Check if the value contains coordinates
                coordinates = value
                left_click_coordinates_raw(coordinates)
            else:  # Otherwise, treat it as an image name
                image_name = value
                left_click_image(image_name)
        elif task.startswith("OOB_TextInput ="):
            text = task.split("=")[1].strip()
            text_input(text)
        elif task.startswith("OOB_SpecialKey ="):
            key = task.split("=")[1].strip()
            special_key(key)
        elif task.startswith("OOB_FunctionKey ="):
            key = task.split("=")[1].strip()
            function_key(key)
        elif task.startswith("OOB_KeyCombo ="):
            combo = task.split("=")[1].strip()
            key_combo(combo)
        elif task.startswith("OOB_Delay ="):
            delay = int(task.split("=")[1].strip())
            delay_task(delay)
        elif task.startswith("OOB_MouseMove ="):
            coordinates = task.split("=")[1].strip()
            move_mouse_to_coordinates_raw(coordinates)
        elif task == "OOB_TakeScreenshot":
            take_screenshot_task()
        elif task == "OOB_PageUp":
            pyautogui.press("pageup")
        elif task == "OOB_PageDown":
            pyautogui.press("pagedown")
        else:
            print(f"Unknown OOB task: {task}")

        time.sleep(0.2)

def execute_oob_tasks_with_delay():
    """
    Executes the saved OOB tasks after a delay.
    """
    time.sleep(0.1)  # Short delay to ensure the LLM's response has completed
    with open(oob_tasks_file, 'r') as f:
        oob_tasks = json.load(f)
    execute_oob_tasks(oob_tasks)


def delay_task(delay):
    """
    Adds a delay in milliseconds.
    """
    time.sleep(delay / 1000.0)

def move_mouse_to_coordinates_raw(coordinates):
    """
    Moves the mouse to the specified raw coordinates on the specified monitor.

    Args:
        coordinates (str): A string containing the monitor index and coordinates in the format "monitor_index,x,y".
    """
    monitor_index, x, y = map(int, coordinates.split(","))
    monitors = screeninfo.get_monitors()

    if monitor_index >= len(monitors):
        raise ValueError(f"Monitor index {monitor_index} is out of range. There are only {len(monitors)} monitors.")

    monitor = monitors[monitor_index]
    monitor_x, monitor_y = monitor.x, monitor.y
    absolute_x = monitor_x + x
    absolute_y = monitor_y + y
    pyautogui.moveTo(absolute_x, absolute_y)
    print(f"Mouse moved to coordinates: ({absolute_x}, {absolute_y}) on monitor {monitor_index}")

def left_click_coordinates_raw(coordinates):
    """
    Moves the mouse to the specified raw coordinates on the specified monitor and performs a left click.

    Args:
        coordinates (str): A string containing the monitor index and coordinates in the format "monitor_index,x,y".
    """
    monitor_index, x, y = map(int, coordinates.split(","))
    monitors = screeninfo.get_monitors()

    if monitor_index >= len(monitors):
        raise ValueError(f"Monitor index {monitor_index} is out of range. There are only {len(monitors)} monitors.")

    monitor = monitors[monitor_index]
    monitor_x, monitor_y = monitor.x, monitor.y
    absolute_x = monitor_x + x
    absolute_y = monitor_y + y
    pyautogui.moveTo(absolute_x, absolute_y)
    pyautogui.click()
    print(f"Mouse clicked at coordinates: ({absolute_x}, {absolute_y}) on monitor {monitor_index}")

def take_screenshot_task():
    """
    Takes a screenshot using the current values from the Gradio interface.
    """
    take_screenshot(global_vars["global_monitor_index"], global_vars["global_text_queries"], global_vars["global_score_threshold"], global_vars["global_vision_model_question"], global_vars["global_full_image_vision_model_question"])

def move_mouse_to_image(image_name):
    """
    Moves the mouse to the center of the box that defines the image.
    """
    results_json_path = "extensions/Lucid_Autonomy/ImageOutputTest/results.json"
    with open(results_json_path, 'r') as f:
        results = json.load(f)

    for result in results:
        if result["image_name"] == image_name:
            coordinates = result["coordinates"]
            monitor = get_monitor_info(MONITOR_INDEX)
            move_mouse_to_coordinates(monitor, coordinates)
            break
    else:
        print(f"Image name '{image_name}' not found in the results.")

def left_click_image(image_name):
    """
    Moves the mouse to the center of the box that defines the image and performs a left click.
    """
    results_json_path = "extensions/Lucid_Autonomy/ImageOutputTest/results.json"
    with open(results_json_path, 'r') as f:
        results = json.load(f)

    for result in results:
        if result["image_name"] == image_name:
            coordinates = result["coordinates"]
            monitor = get_monitor_info(MONITOR_INDEX)
            move_mouse_to_coordinates(monitor, coordinates)
            pyautogui.click()  # Perform a left click
            break
    else:
        print(f"Image name '{image_name}' not found in the results.")

def move_mouse_to_coordinates(monitor, coordinates):
    """
    Moves the mouse cursor to the specified coordinates on the monitor.

    Args:
        monitor (dict): A dictionary containing the monitor's information.
        coordinates (list): A list containing the coordinates [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = coordinates
    monitor_x, monitor_y = monitor.x, monitor.y
    center_x = monitor_x + (x1 + x2) // 2
    center_y = monitor_y + (y1 + y2) // 2
    pyautogui.moveTo(center_x, center_y)
    print(f"Mouse moved to coordinates: ({center_x}, {center_y})")

def get_monitor_info(monitor_index):
    """
    Gets information about the specified monitor.

    Args:
        monitor_index (int): The index of the monitor to get information about.

    Returns:
        dict: A dictionary containing the monitor's information.
    """
    monitors = screeninfo.get_monitors()

    if monitor_index >= len(monitors):
        raise ValueError(f"Monitor index {monitor_index} is out of range. There are only {len(monitors)} monitors.")

    return monitors[monitor_index]

def history_modifier(history):
    """
    Modifies the chat history before the text generation in chat mode begins.
    """
    # Clean the internal history
    history["internal"] = clean_chat_history(history["internal"])
    # Clean the visible history
    history["visible"] = clean_chat_history(history["visible"])

    # Inject the latest results.json into the internal history
    results_json_path = "extensions/Lucid_Autonomy/ImageOutputTest/results.json"
    with open(results_json_path, 'r') as f:
        compressed_results = json.load(f)
    latest_compressed_results = json.dumps(compressed_results, indent=4)
    history["internal"].append(["", f"<START_COMPRESSED_RESULTS>\n{latest_compressed_results}\n<END_COMPRESSED_RESULTS>"])

    return history

def clean_chat_history(history):
    """
    Cleans the chat history by removing all but the latest instance of the contents between the bookends.
    """
    START_BOOKEND = "<START_COMPRESSED_RESULTS>"
    END_BOOKEND = "<END_COMPRESSED_RESULTS>"

    # Find all instances of the bookends
    bookend_indices = []
    for i, entry in enumerate(history):
        if isinstance(entry, list) and len(entry) == 2:
            if START_BOOKEND in entry[1] and END_BOOKEND in entry[1]:
                bookend_indices.append(i)

    # If there are multiple instances, remove all but the last one
    if len(bookend_indices) > 1:
        for i in bookend_indices[:-1]:
            history[i][1] = re.sub(f"{START_BOOKEND}.*?{END_BOOKEND}", "", history[i][1], flags=re.DOTALL)

    return history

def append_compressed_results(output, compressed_results):
    """
    Appends the compressed results to the output.
    """
    START_BOOKEND = "<START_COMPRESSED_RESULTS>"
    END_BOOKEND = "<END_COMPRESSED_RESULTS>"
    output += f"\n{START_BOOKEND}\n{json.dumps(compressed_results, indent=4)}\n{END_BOOKEND}"
    return output

def text_input(text):
    """
    Types the given text, replacing '/n' with newline characters.
    """
    text = text.replace("/n", "\n")
    pyautogui.typewrite(text)

def special_key(key):
    """
    Presses the specified special key.
    """
    if key == "enter":
        pyautogui.press("enter")
    elif key == "space":
        pyautogui.press("space")
    elif key == "backspace":
        pyautogui.press("backspace")
    elif key == "tab":
        pyautogui.press("tab")
    elif key == "escape":
        pyautogui.press("esc")
    elif key == "shift":
        pyautogui.press("shift")
    elif key == "ctrl":
        pyautogui.press("ctrl")
    elif key == "alt":
        pyautogui.press("alt")
    else:
        print(f"Unsupported special key: {key}")

def function_key(key):
    """
    Presses the specified function key.
    """
    if key.startswith("f") and key[1:].isdigit():
        pyautogui.press(key)
    else:
        print(f"Unsupported function key: {key}")

def key_combo(combo):
    """
    Presses the specified key combination.
    """
    keys = combo.split("+")
    for key in keys:
        if key in ["ctrl", "alt", "shift"]:
            pyautogui.keyDown(key)
        else:
            pyautogui.press(key)
    for key in keys:
        if key in ["ctrl", "alt", "shift"]:
            pyautogui.keyUp(key)

if __name__ == "__main__":
    # Launch the Gradio interface with the specified UI components
    gr.Interface(
        fn=ui,
        inputs=None,
        outputs="ui"
    ).launch()

