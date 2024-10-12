import torch
import gradio as gr
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image, ImageDraw
import json
import os
import spaces

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")

@spaces.GPU
def query_image(img, text_queries, score_threshold):
    text_queries = text_queries.split(",")

    size = max(img.shape[:2])
    target_sizes = torch.Tensor([[size, size]])
    inputs = processor(text=text_queries, images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    outputs.logits = outputs.logits.cpu()
    outputs.pred_boxes = outputs.pred_boxes.cpu()
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

    result_labels = []
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    for box, score, label in zip(boxes, scores, labels):
        box = [int(i) for i in box.tolist()]
        if score < score_threshold:
            continue
        result_labels.append([box, text_queries[label.item()]])
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), text_queries[label.item()], fill="red")

    # Save the annotated image for debugging
    img_pil.save("annotated_image.png")

    # Save the JSON output to a file
    json_output_path = "output.json"
    with open(json_output_path, 'w') as f:
        json.dump(result_labels, f)

    return img_pil, json_output_path

demo = gr.Interface(
    query_image,
    inputs=[gr.Image(), "text", gr.Slider(0, 1, value=0.1)],
    outputs=["image", "file"],
    title="Zero-Shot Object Detection with OWLv2",
)
demo.launch()

