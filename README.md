# Lucid_Autonomy (upload in progress)
An extension that lets the AI take the wheel, allowing it to use the mouse and keyboard, recognize UI elements, and prompt itself :3

This extension was written 100% by [Mistral-Large-Instruct-2407](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) quantized to 8-bit precision with [llama.cpp](https://github.com/ggerganov/llama.cpp) locally.  The transcriptions to achieve this are presented here [RECPITS](Insertlater), they are not ordered well and some conversations lead to dead ends, however most of the errors were me misunderstanding something.  This model is great!

The model also wrote the first draft of the readme. 

# Lucid_Autonomy

## Table of Contents
1. [Overview](#overview)
2. [Key Components](#key-components)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Teaching Your AI](#teaching-your-ai)
8. [Examples](#examples)
9. [Contributing](#contributing)
10. [License](#license)

## Overview

Welcome to Lucid_Autonomy! This (still very much expeimental, and only tested on Linux, there may be directory / vs \ issues in the code on windows) extension is designed to enable a Large Language Model (LLM) to interact autonomously with a computer. It leverages various models and tools to detect objects on the screen, process images, and perform actions such as moving the mouse, clicking, typing text, and pressing keys.

The extension is designed to work within the text-generation-webui ecosystem, a powerful web-based interface for running large language models locally. It enhances the capabilities of Oobabooga's [text-generation-webui](https://github.com/oobabooga/text-generation-webui) by allowing the LLM to interact with the user's computer, effectively giving it the ability to perform tasks that would otherwise require human intervention.


## How the Extension Works:

A screenshot is taken (either by the user or by the AI), the screenshot is sent to [owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble) (you can try the other owlv2 models, this one seemed the best), the owlv2 model identifies all of the UI elements of interest in the extension UI field.

The owlv2 model provides the lower left and upper right boundary boxes for the UI elements.  These boxes are cropped out of the screenshot and sent to [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6) (the 4-bit version will work too).  The MiniCPM model is instructed to provide a description of each cropped image.  The instructions are provided in the extension UI field.

In addition to the UI element identification, the extension also has MiniCPM describe the entire screen. The instructions are provided in the extension UI field.

The cropped image names, coordinates (as well as coordinainte child elements), and descriptions are entered into a constantly updating json file.  The contents of this file are sent to your LLM of choice, with the coordinates omitted (looking into chaning this).

The AI uses these data to make decisions about what to do, and where to move the mouse and enter text.

Please read the contents of this repo to fully understand how to use the extension.

### Context Memory Management

The log file is edited upon every interaction from the user and AI, it keeps up to 2 copies of json information about the screen and UI elemements.  This means the AI does not retain the json information permenanently, the AI needs to type text outside of the json bookend characters to retain information in context.

### GPU Utilization

To optimize GPU usage, the extension loads and unloads vision models as needed. This helps manage VRAM utilization, ensuring that the system remains responsive even when running multiple tasks.  (yes, the model loads and unloads one extra time for right now, will fix later)

### Dual Screen Setup

For the best experience, it is recommended to use a dual-screen setup. This allows the textgen UI to run on one screen while the AI interacts with the UI on the other screen. This separation helps in monitoring the AI's actions and providing clear instructions.  If you only have one screen, it is best to cover the textgen UI with an "always on top" notpad or something so the AI isn't reading its text twice essentially.

## Test your setup before involing your LLM

It is important to troubleshoot and understand what the owl2 model can "see" how the threashold values affect the results and such.  I would consider a threashold value of 0.1 low and a value of 0.3 high, with 0.24 being good for most casses.

To do this run the app.py file via the terminal link in the textgen directory for your operating system, for linux it is cmd_linux.sh for example.

With the terminal open run (no quotes in directory name)

```
cd "the Lucid_Autonomy extenssion directory"
python app.py
```

This will yeild the following ui via a hyperlink:

![image](https://github.com/user-attachments/assets/81b13353-2e91-4b39-9e9e-c260595826f7)

here you can test out screenshots you take manually using different descriptors and threshold values.  

The code for this was originally gotten from here: 

https://huggingface.co/spaces/merve/owlv2

and I had Mistral update the code to work offline, use a differnet model, and add more functionality.

## Test your setup with your LLM using the extension, and example of how to use the extenssion.

The repo include a file called "WebColorChange.html" you can run this in a web browser and see if your model is functioning correctly.  The webpage will load white, but will randomly change colors when the correct button is pressed.  Try the following steps to see if things are functioning correctly:

1. Run the the WebColorChange web page:
![image](https://github.com/user-attachments/assets/f8399cc1-f846-4e8f-9317-97fd2dbcbe5e)

2. Press the Take Screenshot button after adjusting the necessary fields.  You will see a number of new files and folders created in the Lucid_Autonomy extenssion folder.
   - screenshot.png is the original screenshot
   - Human_annotated_image.png has red boxes around everything that owlv2 detected, this is good to have open and let auto refresh upon every consecutiative screensshot so you can see what is going on.
   - annotated_image.png is the png that is cropped for every box and sent to the MiniCPM model (I will remove these duplicates later)
   - ImageOutputTest is a folder that gets updated upon every screenshot that has the cropped images and json files.  Depending in how many images there are and what you are having MiniCPM do, this could take a while.
  
   ![image](https://github.com/user-attachments/assets/1927353b-ad01-4c8f-bb3d-9c5650469aa4)


3. Begin talking with your AI, start out with:

   ```
   Can you autonomously change the background color until the background is red and then cease sending yourself tasks?

   0,896,2052 takes you to the text input field for the textgen software we are running, this is where you can click to paste your inner thoughts to progress through the objective autonomously
   ```
   The AI can choose pictures to click on, and it can also use coordinates.  0,896,2052 is monitor=0, x=896, y=2052  The coordinates I provided the AI are for my setup, you need to figure out where pertientn UI elements are for your setup.  To do this I use XnView https://www.xnview.com/en/ and view screenshots in the image viewer, UI shows the x,y coordinates on the screen, most image viewers will tell you the x,y coordinates.  Being able to tell the AI where perminent UI features are is useful.

4. Here is an example of the AI's response:

   ![image](https://github.com/user-attachments/assets/8d3d1861-e5ae-4102-ac02-8a8144181ff6)

   You will notice that the json information is visible, I will change this later, but it will be delted and does not accumulate.
   
   ![image](https://github.com/user-attachments/assets/2e4b34ed-5e94-455f-8d87-913fff18d500)

   I only sent the original message, the AI did the rest on its own.

I've gotten the AI to start on the incognido page, where it figures out where to enter google.com, does a web search for animal facts, chooses a hyperlink, and scroll down the page accumulating information as it reads the page.  

This is a bit slow (doing a page down for every screenshot and hoping MiniCPM will pick up enough text), I will be mashing my LucidWebSearch extension with this extension so the AI can get more complete information on the pages it decides to reads.

If you use the provided character card the AI can be a little eager to do stuff on its own, you can tell the AI to hold off at the beginning of a conversation else you can try the numerous other ways to teach your AI how to use the extension.





## Key Components

### Models
- **Owlv2 Model**: Used for object detection based on text queries.
- **MiniCPM-Llama3 Model**: Used for processing images and generating descriptions.

### Tools
- **Gradio Interface**: Provides a user interface for configuring and triggering actions.
- **PyAutoGUI**: Used for automating mouse and keyboard actions.
- **Screeninfo**: Used for getting information about connected monitors.

### Functions
- **take_screenshot**: Captures a screenshot of a specified monitor.
- **query_image**: Processes the screenshot using the Owlv2 model to detect objects.
- **crop_images**: Crops out detected objects and processes them with the MiniCPM model.
- **process_with_vision_model**: Processes images with the MiniCPM model and returns descriptions.
- **execute_tasks**: Executes tasks listed in the LLM's output, such as moving the mouse, clicking, typing text, and pressing keys.
- **execute_oob_tasks**: Executes out-of-band (OOB) tasks after a delay.

## Features

- **Object Detection**: Detects objects on the screen based on text queries.
- **Image Processing**: Processes images and generates descriptions.
- **Automation**: Automates mouse and keyboard actions.
- **Task Execution**: Executes tasks based on LLM output.
- **Integration**: Seamlessly integrates with the textgen ecosystem.



## Configure Settings

Download the version of MiniCPM-V-2_6 that works for you [Normal Precision](https://huggingface.co/openbmb/MiniCPM-V-2_6) or [4-bit Precision](https://huggingface.co/openbmb/MiniCPM-V-2_6-int4), (this code was developed around v2_5 but 2_6 just came out and seems to function better).

Edit the script.py file VISION_MODEL_ID variable in any text editor with the directory of your downloaded model:

![image](https://github.com/user-attachments/assets/b120b6ab-92cf-4d1a-890c-ec71ff26c0fb)

```
VISION_MODEL_ID = "enter your directory here"
```
Here is a screenshot of the UI:

![image](https://github.com/user-attachments/assets/79aea1fd-719b-49ec-acb0-75283afb99ad)

   - **Monitor Index**: Select the monitor to capture. Change default in code if you are having issues.
   - **Text Queries**: Enter comma-separated text queries for object detection.  Do not allow for spaces before or after commas.  This is the query that is going be sent to the OWLv2 Model.  This model has a very interesting prompting style that has a lot of potential.  For example adding "x" without the qutoes, in the list of text queires and a low threahold value of 0.1 will find all the close window Xs, similarily searcing for "speech bubble" with a low threashold value of 0.1 will find all of the reply icons on a reddit page.

![image](https://github.com/user-attachments/assets/cbacbfde-bbf9-454f-aa15-ed5ffefcccc6)

     
   - **Score Threshold**: Set the score threshold for filtering low-probability predictions.
   - **Vision Model Question**: Enter the question to ask the vision model for cropped images.
   - **Full Image Vision Model Question**: Enter the question to ask the vision model for the full image.

3. **Take Screenshot**:
   - Click the "Take Screenshot" button to capture a screenshot and process it.

### Workflow

1. **Capture Screenshot**: The AI captures a screenshot of the specified monitor.
2. **Object Detection**: The Owlv2 model detects objects based on text queries.
3. **Image Processing**: The MiniCPM model processes the detected objects and generates descriptions.
4. **Task Execution**: The AI executes tasks based on the LLM's output.

## Configuration

### Configurable Variables

- **MONITOR_INDEX**: Index of the monitor to capture (0 for the first monitor, 1 for the second monitor, etc.).
- **TEXT_QUERIES**: Comma-separated text queries for object detection.
- **SCORE_THRESHOLD**: Score threshold for filtering low-probability predictions.
- **VISION_MODEL_QUESTION**: Question to ask the vision model for cropped images.
- **FULL_IMAGE_VISION_MODEL_QUESTION**: Question to ask the vision model for the full image.

### Customization

You can customize the extension by modifying the configurable variables and settings in the code. This allows you to tailor the extension to your specific needs.

## Teaching Your AI

There are several ways to teach your AI how to use the Lucid_Autonomy extension:

1. **Use the Included Character Card File**:
   - The repository includes a character card file that contains a detailed instruction set. This file is sent to the AI upon every back-and-forth between the user and the AI.

2. **Create Your Own Character Card**:
   - You can create your own character card by writing a block of text that contains the instructions you want the AI to follow.

3. **Combine Character Card Information with In-Conversation Instructions**:
   - You can paste the character card information into the conversation with the AI and/or use a character card. This approach allows you to provide additional context and instructions as needed.

4. **Teach Your AI Step-by-Step**:
   - You can teach your AI like a person, step by step. For example, you can show the AI how to move the mouse and use the trigger phrase. The AI can then build upon this knowledge to perform other tasks. This method is particularly useful for tasks that don't require complex instructions.

### Explore and Experiment

Encourage your AI to explore and experiment with different teaching methods. You might find that a combination of approaches works best for your specific use case.

## Examples

### Example 1: Moving the Mouse

1. **Teach the AI to Move the Mouse**:
   - Instruct the AI to move the mouse to a specific location on the screen.
   - Use the trigger phrase "MouseMover = image_name" to execute the task.

2. **AI's Response**:
   - The AI will move the mouse to the center of the box that defines the image.

### Example 2: Clicking a Button

1. **Teach the AI to Click a Button**:
   - Instruct the AI to click a button on the screen.
   - Use the trigger phrase "MouseClick = image_name" to execute the task.

2. **AI's Response**:
   - The AI will move the mouse to the center of the box that defines the image and perform a left click.

### Example 3: Typing Text

1. **Teach the AI to Type Text**:
   - Instruct the AI to type a specific text.
   - Use the trigger phrase "TextInput = text" to execute the task.

2. **AI's Response**:
   - The AI will type the specified text, replacing "/n" with newline characters.

## Contributing

We welcome contributions from the community! If you have any ideas, suggestions, or bug reports, please feel free to open an issue or submit a pull request.

### Guidelines

1. **Fork the Repository**:
   - Fork the repository and create a new branch for your changes.

2. **Make Your Changes**:
   - Make your changes and ensure that they are well-documented.

3. **Submit a Pull Request**:
   - Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

Please review this README and let me know if you'd like to make any further changes or additions.
