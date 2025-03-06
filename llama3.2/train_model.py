import os
import torch
import pickle as pkl
import pandas as pd
from PIL import Image
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig

#############################
# 1. Load the model
#############################
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Fixed instruction prompting the model for a short answer.
instruction = (
    "You are an expert neurologist. Based on your observations, indicate if the patient is likely to have: "
    "either alzheimers, mild cognitive impairment, or none. Answer '0' corresponds to none. Answer '1' corresponds to mild cognitive impairment. Answer '2' corresponds to alzheimers."
)

#########################################
# 2. Load training images and labels
#########################################
# Load images from pickle (assuming a DataFrame-like structure with column 'im1')
with open("img_train.pkl", "rb") as f:
    train_images_data = pkl.load(f)
df_train = pd.DataFrame(train_images_data)
train_images = [row["im1"] for _, row in df_train.iterrows()]

# Load corresponding labels (a list of strings, one per image)
with open("img_y_train.pkl", "rb") as f:
    train_labels = pkl.load(f)
train_labels = train_labels['label']
def convert_to_conversation(image, label):
    """
    Converts an image (in PIL format) and its label into a conversation sample.
    The user message contains the instruction and the image,
    and the assistant message provides the short label.
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": label}
            ]
        }
    ]
    return {"messages": conversation}

# Create the training dataset by zipping images and labels.
converted_dataset = [
    convert_to_conversation(img, label)
    for img, label in zip(train_images, train_labels)
]

#########################################
# 3. Quick Inference Test (Before Training)
#########################################
FastVisionModel.for_inference(model)
test_image_example = train_images[0]  # just a quick test using the first image

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    test_image_example,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

print("\nBefore training (example generation):\n")
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=10,
    use_cache=True,
    temperature=1.5,
    min_p=0.1
)

#########################################
# 4. Training
#########################################
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=10,  # For demonstration; adjust for full training runs
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=2048,
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / (1024 ** 3), 3)
max_memory = round(gpu_stats.total_memory / (1024 ** 3), 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / (1024 ** 3), 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

#########################################
# 5. Few-Shot Evaluation on Test Set
#########################################
# Load test images and labels
with open("img_test.pkl", "rb") as f:
    test_images_data = pkl.load(f)
df_test = pd.DataFrame(test_images_data)
test_images = [row["im1"] for _, row in df_test.iterrows()]

with open("labels_test.pkl", "rb") as f:
    test_labels = pkl.load(f)

def create_few_shot_prompt(test_image, num_shots, demo_dataset):
    """
    Constructs a prompt with num_shots demonstration examples (each including both user and assistant turns)
    followed by the test sample prompt (only the user turn).
    """
    messages = []
    # Use the first num_shots examples from the training set as demonstrations
    for example in demo_dataset[:num_shots]:
        messages.extend(example["messages"])
    # Add the test sample prompt
    test_prompt = {
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            {"type": "image", "image": test_image}
        ]
    }
    messages.append(test_prompt)
    return messages

num_shots = 3  # You can change this parameter to evaluate 1-shot, 3-shot, etc.
predictions = []
model.eval()
with torch.no_grad():
    for test_image, true_label in zip(test_images, test_labels):
        prompt_messages = create_few_shot_prompt(test_image, num_shots, converted_dataset)
        input_text = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)
        inputs = tokenizer(
            test_image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")
        # Use a low temperature (or even 0) for deterministic output
        output_ids = model.generate(**inputs, max_new_tokens=10, use_cache=True, temperature=0.0)
        pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()
        predictions.append((pred_text, true_label.lower()))

# Compute accuracy (exact match)
correct = sum(1 for pred, true in predictions if pred == true)
accuracy = correct / len(predictions) * 100
print(f"\nFew-shot evaluation with {num_shots} shot(s): Accuracy = {accuracy:.2f}%")

#########################################
# 6. Save the model
#########################################
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")