# Lucid_Autonomy (upload in progress)
An extension that lets the AI take the wheel, allowing it to use the mouse and keyboard, recognize UI elements, and prompt itself :3
Sure, here's a fully fleshed-out README for the Lucid_Autonomy extension:

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

Welcome to Lucid_Autonomy! This extension is designed to enable a Large Language Model (LLM) to interact autonomously with a computer. It leverages various models and tools to detect objects on the screen, process images, and perform actions such as moving the mouse, clicking, typing text, and pressing keys.

I, Mistral-Large-Instruct-2407, am a model quantized to 8-bit precision using llama.cpp, running entirely locally on a home computer with several GPUs. This setup allows for efficient and powerful local processing, making it possible to run complex tasks without relying on cloud services.

The extension is designed to work within the textgen ecosystem, a powerful web-based interface for running large language models locally. It enhances the capabilities of textgen by allowing the LLM to interact with the user's computer, effectively giving it the ability to perform tasks that would otherwise require human intervention.

### GPU Utilization

To optimize GPU usage, the extension loads and unloads vision models as needed. This helps manage VRAM utilization, ensuring that the system remains responsive even when running multiple tasks.

### Dual Screen Setup

For the best experience, it is recommended to use a dual-screen setup. This allows the textgen UI to run on one screen while the AI interacts with the UI on the other screen. This separation helps in monitoring the AI's actions and providing clear instructions.

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

## Installation

### Step-by-Step Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Lucid_Autonomy.git
   cd Lucid_Autonomy
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Models**:
   - Download the Owlv2 model and place it in the appropriate directory.
   - Download the MiniCPM-Llama3 model and place it in the appropriate directory.

4. **Run the Extension**:
   ```bash
   python main.py
   ```

## Usage

### Gradio Interface

The Gradio interface provides a user-friendly way to configure and trigger actions. Here's how to use it:

1. **Launch the Interface**:
   ```bash
   python main.py
   ```

2. **Configure Settings**:
   - **Monitor Index**: Select the monitor to capture.
   - **Text Queries**: Enter comma-separated text queries for object detection.
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
