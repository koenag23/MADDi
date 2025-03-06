"""
To install Unsloth:

conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

To open Unsloth Venv:
conda activate unsloth_env

As for how the files are supposed to be, make sure the test and train files are in the same directory as train.py
"""

from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
import torch
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig
import pickle as pkl
import pandas as pd

def convert_to_conversation(df):
    with open(path, 'rb') as f:
        object = pkl.load(f)
    
    df = pd.DataFrame(object)
    
    messages = []
    
    for sindex in range(df.shape[0]):
        row = df.iloc[sindex]
        
        axial = row['im1'].convert('L')
        coronal = row['im2'].convert('L')
        sagittal = row['im2'].convert('L')
        subject = row['subject']
        visit = row['visit']
        
        caption = f"This is an Axial MRI Scan of {subject}'s {visit} visit."
        
        instruction = "You are an expert radiographer. Classify this image as Cognitively Normal (CN), Mild Cognitive Impairment (MCI), or Dementia (D)"
        
        conversation = [
            { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : instruction},
                {"type" : "image", "image" : axial} ]
            },
            { "role" : "assistant",
            "content" : [
                {"type" : "text",  "text"  : caption} ]
            },
        ]
        
        messages.append({ "messages" : conversation })
        
        caption = f"This is a Coronal MRI Scan of {subject}'s {visit} visit."
        
        conversation = [
            { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : instruction},
                {"type" : "image", "image" : coronal} ]
            },
            { "role" : "assistant",
            "content" : [
                {"type" : "text",  "text"  : caption} ]
            },
        ]
        
        messages.append({ "messages" : conversation })
        
        caption = f"This is a Sagittal MRI Scan of {subject}'s {visit} visit."
        
        conversation = [
            { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : instruction},
                {"type" : "image", "image" : sagittal} ]
            },
            { "role" : "assistant",
            "content" : [
                {"type" : "text",  "text"  : caption} ]
            },
        ]
        
        messages.append({ "messages" : conversation })
        
    return messages

path = 'img_train.pkl'
    
messages = convert_to_conversation(path)

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    device_map = "cuda:0",
    use_exact_model_name=True
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

# Still working on getting functional dataset
path = 'img_test.pkl'
with open(path, 'rb') as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
messages = convert_to_conversation(path)

#altered to incorporate all 3 types of scans for a more well-rounded prediction
image1 = df['im1'][0].convert('L')
image2 = df['im2'][0].convert('L')
image3 = df['im3'][0].convert('L')

metadata_fields = ["subject", "visit"] #altered as per metadata fields in img_test
metadata = {field: df[field][0] for field in metadata_fields if field in df}

metadata_text = "\n".join([f"{key}: {value}" for key, value in metadata.items()])

#prompt with mri + metadata input
instruction = (
    "You are a expert radiologist specializing in neuroimaging. "
    "Analyze the provided brain MRI scans, which include axial, coronal, and sagittal views, along with their corresponding metadata. "
    "Based on the imaging features and patient information, classify the case into one of the following categories: "
    "Cognitively Normal (CN), Mild Cognitive Impairment (MCI), or Dementia (D).\n\n"
    f"Patient Metadata:\n{metadata_text}"

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image1},
        {"type": "image", "image": image2},
        {"type": "image", "image": image3},
        {"type": "text", "text": instruction}
    ]}
]

""" FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
) """

FastVisionModel.for_inference(model) # Enable for inference!
input_text = tokenizer.apply_chat_template(messages, chat_template=None, add_generation_prompt = True)
inputs = tokenizer(
    [image1, image2, image3], #model will incorporate all 3 types of scans rather than just im1
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)

