# Lucid_Autonomy (an experiment in progress, not all features are documented)

#### Small bug right now, you need to hard code which monitor the LLM has control over.  The change to the monitor in the Lucid_Autonomy extension UI element in textgen will not correctly be applied to the LLM moving the mouse. 

In the script.py file change the MONITOR_INDEX value to the monitor you want the LLM to control, for me my main monitor is 0 and my smaller monitor is 1.
```
# Configurable variables
MONITOR_INDEX = 1  # Index of the monitor to capture (0 for the first monitor, 1 for the second monitor, etc.)
```
Updates coming! 

![image](https://github.com/user-attachments/assets/b4a56678-0153-4e9b-9ff6-066b28c84123)

![image](https://github.com/user-attachments/assets/6d2871b1-b512-41d6-88e6-e72b3b2576c1)
![image](https://github.com/user-attachments/assets/a26ee978-9cf8-46e2-97cf-9c41ed8023a7)

Why go process with so many models?

Let's look at an example of a bar chart where both GOT-OCR and Marker fail in reproducing the data from the chart:

![image](https://github.com/user-attachments/assets/fefc53fc-8dbc-48ea-b3a0-bea91baf9363)
![image](https://github.com/user-attachments/assets/fa18d368-c03f-4057-8b87-b9f5def57561)
![image](https://github.com/user-attachments/assets/bc8f59c3-a360-4d0a-88bc-f4da08bbc865)
![image](https://github.com/user-attachments/assets/1ee8d739-0659-47ff-abe5-651dc738c63c)



An extension that lets the AI take the wheel, allowing it to use the mouse and keyboard, recognize UI elements, and prompt itself :3

This extension was written 100% by [Mistral-Large-Instruct-2407](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) quantized to 8-bit precision with [llama.cpp](https://github.com/ggerganov/llama.cpp) locally.  The transcriptions to achieve this are presented here [RECEIPTS](https://github.com/RandomInternetPreson/Lucid_Autonomy/tree/main/Receipts), they are not ordered well and some conversations lead to dead ends, however most of the errors were me misunderstanding something.  This model is great!

The model also wrote the first draft of the readme. 

#### todo:
- add double click
- integrate lucid vision multi model selection seheme
- integrate lucid websearch for AI to utilize on its own
- develop better "spatial awarness" json scheme, I have several ideas on how to contextualize the UI element spatial data better in the json file, the current method is just okay
- try out copy all and past all techniques for quick text input into model
- try out dual models clicking into each others text fields and communicating

# Lucid_Autonomy

## Table of Contents
1. [Overview](#overview)
2. [How the Extension Works](#how-the-extension-works)
3. [Test your setup before involving your LLM](#Test-your-setup-before-involving-your-LLM)
4. [Test your setup with your LLM using the extension, and example of how to use the extension](#test-your-setup-with-your-llm-using-the-extension-and-example-of-how-to-use-the-extension-test-with-llama-31-70b-converted-into-an-8-bit-guff)
5. [Teaching Your AI](#teaching-your-ai)
6. [Understanding the speficificss of how the extension works](#how-to-understand-the-specifics-of-what-the-extension-can-do-and-how-to-properly-format-tasks)
7. [Configure Settings](#tconfigure-settings)
9. [Tips](#Tips)
10. [Contributing](#Contributing)

# Overview 

Welcome to Lucid_Autonomy! This (still very much experimental, and only tested on Linux, there may be directory / vs \ issues in the code on windows) extension is designed to enable a Large Language Model (LLM) to interact autonomously with a computer. It leverages various models and tools to detect objects on the screen, process images, and perform actions such as moving the mouse, clicking, typing text, and pressing keys.

The extension is designed to work within the text-generation-webui ecosystem, a powerful web-based interface for running large language models locally. It enhances the capabilities of Oobabooga's [text-generation-webui](https://github.com/oobabooga/text-generation-webui) by allowing the LLM to interact with the user's computer, effectively giving it the ability to perform tasks that would otherwise require human intervention.

If you use the included AI character card you will need a model with a lot of context and have enough vram to support a lot of context; with a minimum of around 60k tokens. 

You can teach in context and/or create your own character card with far smaller context requriements; discussed below.


# How the Extension Works

A screenshot is taken (either by the user or by the AI), the screenshot is sent to [owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble) (you can try the other owlv2 models, this one seemed the best), the owlv2 model identifies all of the UI elements of interest in the extension UI field.

The owlv2 model provides the lower left and upper right boundary boxes for the UI elements.  These boxes are cropped out of the screenshot and sent to [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6) (the 4-bit version will work too).  The MiniCPM model is instructed to provide a description of each cropped image.  The instructions are provided in the extension UI field.

In addition to the UI element identification, the extension also has MiniCPM describe the entire screen. The instructions are provided in the extension UI field.

The cropped image names, coordinates (as well as coordinate child elements), and descriptions are entered into a constantly updating json file.  The contents of this file are sent to your LLM of choice, with the coordinates omitted (looking into changing this).

The AI uses these data to make decisions about what to do, and where to move the mouse and enter text.

Please read the contents of this repo to fully understand how to use the extension.

### Workflow

1. **Capture Screenshot**: The AI captures a screenshot of the specified monitor.
2. **Object Detection**: The Owlv2 model detects objects based on text queries.
3. **Image Processing**: The MiniCPM model processes the detected objects and generates descriptions.
4. **Task Execution**: The AI executes tasks based on the vision model outputs.

### Context Memory Management

The log file is edited upon every interaction from the user and AI, it keeps up to 2 copies of json information about the screen and UI elements.  This means the AI does not retain the json information permanently, the AI needs to type text outside of the json bookend characters to retain information in context.  Install the Model Ducking extension if you only have one gpu or want to offload your model while the vision models work https://github.com/BoredBrownBear/text-generation-webui-model_ducking

### GPU Utilization

To optimize GPU usage, the extension loads and unloads vision models as needed. This helps manage VRAM utilization, ensuring that the system remains responsive even when running multiple tasks.  (yes, the model loads and unloads one extra time for right now, will fix later)

You can change the gpu you want the model to load on by changing the value of device near the topish of the code:

```
device = 0
```

### Dual Screen Setup

For the best experience, it is recommended to use a dual-screen setup. This allows the textgen UI to run on one screen while the AI interacts with the UI on the other screen. This separation helps in monitoring the AI's actions and providing clear instructions.  If you only have one screen, it is best to cover the textgen UI with an "always on top" notepad or something so the AI isn't reading its text twice essentially.

# Test your setup before involving your LLM

It is important to troubleshoot and understand what the owl2 model can "see" how the threshold values affect the results and such.  I would consider a threshold value of 0.1 low and a value of 0.3 high, with 0.24 being good for most cases.

To do this run the app.py file via the terminal link in the textgen directory for your operating system, for linux it is cmd_linux.sh for example.

With the terminal open run (no quotes in directory name)

```
cd "the Lucid_Autonomy extension directory"
python app.py
```

This will yeild the following ui via a hyperlink:

![image](https://github.com/user-attachments/assets/81b13353-2e91-4b39-9e9e-c260595826f7)

here you can test out screenshots you take manually using different descriptors and threshold values.  

The code for this was originally gotten from here: 

https://huggingface.co/spaces/merve/owlv2

and I had Mistral update the code to work offline, use a different model, and add more functionality.

# Test your setup with your LLM using the extension, and example of how to use the extension (test with Llama-3.1-70B converted into an 8-bit .guff)

The repo includes a file called "WebColorChange.html" you can run this in a web browser and see if your model is functioning correctly.  The webpage will load white, but will randomly change colors (through a finite list of colors) when the correct button is pressed.  Try the following steps to see if things are functioning correctly:

1. Run the the WebColorChange web page:
![image](https://github.com/user-attachments/assets/f8399cc1-f846-4e8f-9317-97fd2dbcbe5e)

2. Press the Take Screenshot button after adjusting the necessary fields.  You will see a number of new files and folders created in the Lucid_Autonomy extension folder.
   - screenshot.png is the original screenshot
   - Human_annotated_image.png has red boxes around everything that owlv2 detected, this is good to have open and let auto refresh upon every consecutive screenshot so you can see what is going on.
   - annotated_image.png is the png that is cropped for every box and sent to the MiniCPM model (I will remove these duplicates later)
   - ImageOutputTest is a folder that gets updated upon every screenshot that has the cropped images and json files.  Depending on how many images there are and what you are having MiniCPM do, this could take a while.
     
   ![image](https://github.com/user-attachments/assets/0ec35a76-4996-4c75-8c07-495ddedbe148)

  
  
   ![image](https://github.com/user-attachments/assets/1927353b-ad01-4c8f-bb3d-9c5650469aa4)


3. Begin inferenceing with your AI, start out with:

   ```
   Can you autonomously change the background color until the background is red and then cease sending yourself tasks?

   0,896,2052 takes you to the text input field for the textgen software we are running, this is where you can click to paste your inner thoughts to progress through the objective autonomously
   ```
   The AI can choose pictures to click on, and it can also use coordinates.  0,896,2052 is monitor=0, x=896, y=2052  The coordinates I provided the AI are for my setup, you need to figure out where pertinent UI elements are for your setup.  To do this I use XnView https://www.xnview.com/en/ and view screenshots in the image viewer, the UI shows the x,y coordinates on the screen, most image viewers will tell you the x,y coordinates.  Being able to tell the AI where pertinent UI features are is useful.

4. Here is an example of the AI's response:

   ![image](https://github.com/user-attachments/assets/8d3d1861-e5ae-4102-ac02-8a8144181ff6)

   You will notice that the json information is visible, I will change this later, but it will be delted and does not accumulate.
   
   ![image](https://github.com/user-attachments/assets/2e4b34ed-5e94-455f-8d87-913fff18d500)

   I only sent the original message, the AI did the rest on its own.

I've gotten the AI to start on the incognito page, where it figures out where to enter google.com, does a web search for animal facts, chooses a hyperlink, and scroll down the page accumulating information as it reads the page.  

This is a bit slow (doing a page down for every screenshot and hoping MiniCPM will pick up enough text), I will be mashing my LucidWebSearch extension with this extension so the AI can get more complete information on the pages it decides to reads.

If you use the provided character card the AI can be a little eager to do stuff on its own, you can tell the AI to hold off at the beginning of a conversation else you can try the numerous other ways to teach your AI how to use the extension.

# Teaching Your AI

There are several ways to "teach" your AI how to use the Lucid_Autonomy extension:

1. **Use the Included Character Card File**:
   - The repository includes a character card file that contains a detailed instruction set. This file is sent to the AI upon every back-and-forth between the user and the AI.  This character card is long...like really long; it is likely longer than it needs to be.  You will need to edit your setting.yaml file for textgen to accommodate more token than the max of 4096, I set mine to 10240 (I'm not 100% sure this is necessary, but it seems to help):
     ```
     prompt-default: None
     prompt-notebook: None
     preset: Debug-deterministic
     max_new_tokens: 10240
     truncation_length: 131072
     custom_stopping_strings: '"<START_COMPRESSED_RESULTS>"'
     character: AI
     ```

'"<START_COMPRESSED_RESULTS>" is the beginning of the bookend text that straddles the json data inside the chat log, sometimes models want to print it out, not often but sometimes.  In addition, if you use the character card, you may want to change the coordinates in examples into your own coordinates.
     
2. **Create Your Own Character Card - can be good for less context too**:
   - You can create your own character card by writing a block of text that contains the instructions you want the AI to follow.  Take a look at the current character card to see what is available to the model, there are more features available to the model than are what in the current character card.

3. **Combine Character Card Information with In-Conversation Instructions - can be good for less context too**:
   - You can paste the character card information into the conversation with the AI and/or use a character card. This approach allows you to provide additional context and instructions as needed.

4. **Teach Your AI Step-by-Step - can be good for less context too**:
   - You can teach your AI like a person, step by step in context. For example, you can show the AI how to move the mouse and use the trigger phrase. The AI can then build upon this knowledge to perform other tasks. This method is particularly useful for tasks that don't require complex instructions or something you want the AI to repeat.  This is how I started, I taught the AI step by step as the code was being developed making sure that the code worked and that the AI could use the tools correctly.  You may find that you need to teach your AI like a person unfamiliar with a computer, which might seem odd; but from the models I've tried LLMs seem to lack this knowledge intrinsically. However, they can grasp the logic in context.

### Explore and Experiment

Encourage your AI to explore and experiment with different teaching methods. You might find that a combination of approaches works best for your specific use case.

# How to understand the specifics of what the extension can do and how to properly format tasks

Regardless of your intent to use the included "AI" character card, it is a good idea to read the entire thing to get an understanding of what the AI can do and how it should format tasks.  I will highlight some main passages here:

```
- The AI has access to the following tools: OOB_MouseClick, OOB_TextInput, OOB_SpecialKey, OOB_MouseMover, OOB_MouseMove, OOB_Delay, OOB_TakeScreenshot, OOB_PageUp, and OOB_PageDown.

- OOB_MouseClick is used in the following manners:

OOB_MouseClick = "image_name"

OR

OOB_MouseClick = Monitor_Number,x,y

When the AI adds OOB_MouseClick to the task list and provides an image name from the most recent version of the compressedresults.json file it has in context (without the quotes and without a Monitor_Number), the Lucid_Autonomy extension will left click the mouse button at the x,y coordinates determined by the OWLv2 model for that "specific_image."  Thus the AI should study the compressedresutls.json file to help it make decisions about actions to take.  The  image_name “description” is the only way the AI can know which UI element to interact with.
```

```
- Here are some tasks with reasoning between tasks, putting reasoning between tasks is not the proper format, but this is to help the AI understand how to effectively use a computer, which is ironic because the AI is a computer:

Task 1: OOB_MouseClick = text bar field_17.png

The first task is to click on the text bar field with the description "Search icon with microphone and settings options." This is the search bar on the Google homepage, where the AI wants to enter our search query.  This image is inside of a larger image through its "children" association in the json file, thus this is the more likely image to select even though there is another image with a similar description.

Task 2: OOB_TextInput = "latest advancements in space exploration"

The second task is to enter the search query "latest advancements in space exploration" into the search bar. This will allow the AI to search for information on this topic.

Task 3: OOB_SpecialKey = enter

The third task is to press the enter key, which will submit the search query and load the search results page.

Task 4: OOB_Delay = 4000

The fourth task is to wait for 4 seconds, which will give the page time to load and the search results to appear.

Task 5: OOB_TakeScreenshot

The fifth task is to take a screenshot of the current page, which will capture the search results and any other information on the page.

Task 6: OOB_MouseClick = 0,896,2052

The sixth task is to click on the textgen text input field on monitor 0, which is where the AI will enter its out-of-body message to itself.

Task 7: OOB_TextInput = I just took a screenshot of the search results page for latest advancements in space exploration. I'm going to choose a link to click on and continue exploring, note any interesting information or text on the screen like a person would do if they were performing a search for information using the description for full_image.png.  If the page is not as expected I'll revaluate and choose the best set of actions. If the full_image.png description indicates I am looking the search results page I expected to be on, I'll write out detailed observations and records from the contents of the full_image.png description.  I will create a separate task list every time I wake up, and decide to continue documenting the page contents or to conclude my task is complete and cease creating a task list. I will identify my reasoning.

The seventh task is to paste the out-of-body message to the AI, which will help the AI keep track of what the AI is doing and to plan the AI's next move.

Task 8: OOB_SpecialKey = enter

The eighth task is to press enter, which will submit the OOB_TextInput text to the textgen program, this will allow the AI to "wake up" and continue contextualizing the screenshot information against its goals so it can dynamically react to the changing screenshot information.

By understanding the correct sequence of tasks and the role of the OOB_Delay, the AI can see that the OOB_TakeScreenshot task is the last action the AI will take while "awake" before the AI's turn is over. The OOB tasks are then executed while the AI is "asleep", allowing the AI to enter its out-of-body message and submit the text which will "wake up" the AI allowing it to continue exploring the page.
```

```
Out-Of-Body (OOB) Tasks

*The following information is intended to explain to the AI what Out-Of-Body (OOB) tasks are and how to use them within the context of the Lucid_Autonomy extension.

- For the AI to fully comprehend how to use OOB tasks, it is important for the AI to contextualize how the user and AI inferencing execution works. The AI is operating within a framework called textgen, which allows users to run large language models (LLMs) on their personal computer hardware. When a user begins a textgen session, they are presented with the last reply from the AI. If it is a new conversation, a pre-programmed initial greeting is generated as if it came from the AI. This means that the user needs to initiate communication with the AI model.

The user types their message to the AI and then sends the message (this is when the AI receives the updated screenshot information). The AI receives the message and the current screenshot information, formulates its response, and then sends the message to the user.

When the AI has sent its message to the user, the AI goes into a type of sleep. The AI normally stays asleep until it receives a new message from the user. OOB tools and tasks are intended to operate when the AI is asleep.  Hence the Out-Of-Body phrasing, the tools in the task list operate outside the body of the AI, when the AI is not able to be active.

- It is important for the AI to understand how to use OOB tasks and for what purposes.

When given the x,y coordinates for the textgen user input text field, the AI can enter text into that field and press "enter" to submit the text to itself, replacing the role of the user with a form of "inner thought."

The concept of these inner thoughts is very important for the AI to understand and leverage, because it allows the AI to "think ahead" and tell itself to do things when it wakes up from being asleep using cleverly constructed Autonomy_Task lists.

While the AI is formulating a message, it may type out any type of text. The AI may then end the message with the trigger phrase (Autonomy_Tasks:) and a list of tasks; always making sure to end a message with the task list with regard to message formatting. If those tasks involve constructing an "inner thought" via the OOB_TextInput tool, then the AI should realize that these are inner thoughts intended to be sent to itself while asleep (after the AI has sent its message). The AI waits until it gets a response that is only the AI's inner thought without the context of any other text, then the AI acts upon those inner thoughts upon waking up.

Upon waking up the AI will contextualize the received inner thoughts within the context of the most current screenshot information, the AI writes out descriptions of the current screenshot, the AI writes out differences between current and past screenshot description, the AI writes out how the current context and differences change the construction of a new task list if applicable.

The AI will realize when a task is complete and the AI will cease to send itself inner thoughts and task lists when a task is complete.

This is how the inner thoughts allow the AI to think ahead and perform tasks autonomously. They allow the AI to take actions upon waking up. When the AI wakes up, it is provided the updated compressedresults.json file information and the AI can adapt dynamically to this information autonomously. The AI can execute the inner thoughts with updated information about the screen.  The AI will take screenshots as it is a means for the AI to see if intended actions have been executed, by comparing the contents of the new screenshot to that of previous screenshot information.
```

```
- The AI is scrolling through a webpage and takes a screenshot after each scroll. The AI should compare the current screenshot with the previous one to determine if new content has been loaded.

Autonomy_Tasks:
OOB_PageDown
OOB_TakeScreenshot
OOB_MouseClick = 0,896,2052
OOB_TextInput = Inner thought: I just scrolled down the page and took a screenshot. I will continue scrolling and sending myself inner thoughts to think ahead. I will continue this process until I reach the bottom of the page while contextualizing updated page content.  When I see the contents of the new screenshot I will make a comparison between the new screenshot and the previous screenshot. I should be seeing contents of the page change from the full_image.png description.  If the page is not changing I will conclude that I have reached the bottom of the page, and cease to type out task lists, I will identify my reasoning and provide quotes from the full_image description to justify my actions.
OOB_SpecialKey = enter
```

# Configure Settings

Download the version of MiniCPM-V-2_6 that works for you [Normal Precision](https://huggingface.co/openbmb/MiniCPM-V-2_6) or [4-bit Precision](https://huggingface.co/openbmb/MiniCPM-V-2_6-int4), (this code was developed around v2_5 but 2_6 just came out and seems to function better).

#### Small bug right now, you need to hard code which monitor the LLM has control over.  The change to the monitor in the Lucid_Autonomy extension UI element in textgen will not correctly be applied to the LLM moving the mouse. 

In the script.py file change the MONITOR_INDEX value to the monitor you want the LLM to control, for me my main monitor is 0 and my smaller monitor is 1.
```
# Configurable variables
MONITOR_INDEX = 1  # Index of the monitor to capture (0 for the first monitor, 1 for the second monitor, etc.)
```

Edit the script.py file VISION_MODEL_ID variable in any text editor with the directory of your downloaded model:

![image](https://github.com/user-attachments/assets/b120b6ab-92cf-4d1a-890c-ec71ff26c0fb)

```
VISION_MODEL_ID = "enter your directory here"
```
Here is a screenshot of the UI:

![image](https://github.com/user-attachments/assets/79aea1fd-719b-49ec-acb0-75283afb99ad)

   - **Monitor Index**: Select the monitor to capture. Change default in code if you are having issues.
   - **Text Queries**: Enter comma-separated text queries for object detection.  Do not allow for spaces before or after commas.  This is the query that is going be sent to the OWLv2 Model.  This model has a very interesting prompting style that has a lot of potential.  For example adding "x" without the qutoes, in the list of text queires and a low threahold value of 0.1 will find all the close window Xs, similarily searcing for "speech bubble" with a low threshold value of 0.1 will find all of the reply icons on a reddit page.
     * It should be noted however, that decreasing the threshold increases the number of UI elements seen by the owl model which will slow things down.

![image](https://github.com/user-attachments/assets/cbacbfde-bbf9-454f-aa15-ed5ffefcccc6)

     
   - **Score Threshold**: Set the score threshold for filtering probability predictions.
   - **Vision Model Question**: Enter the question to ask the MiniCPM vision model for cropped images.
   - **Full Image Vision Model Question**: Enter the question to ask the MiniCPM vision model for the full image description.
   - **Take Screenshot**: Click the "Take Screenshot" button to capture a screenshot and process it.



# Tips:
- I've found that for the model to active the window with the mouse to page down, that just simple to tell the model that 1,0,0 (monitor 1, x=0, y=0) is where it needs to click to activate the full screen web browser.
- Change the example coordinates from the character card into your own coordinates, this way the AI can recall them better.
- Some AIs do not need to send themselves inner thoughts, they can just click the generate button on their own and keep on doing things, however this behaior seems less stable?
- I know models are not conscious, it is just easier to anthropomorphize them sometimes to integrate them with the extension.  You will find that different AIs get tripped up on the most odd logic, showing how much they do not "reason" or how alien “their” logic is.
- You can customize the extension by modifying the configurable variables and settings in the code. This allows you to tailor the extension to your specific needs.
- There is still a lot of functionality I'm going to add, such as double clicking, and I haven't fully explored all the tools myself.  But this is a place I can better update and organize my results and code.
- There is a lot of vestigial code that I need to remove. Will do later.

- ## Contributing

Feel free to improve the code, I'll be working on better instructions (shorter more concise) and exploring differnet vision model options.

Please consider contributing to oobabooga, they provide a valuable tool to the community :3
