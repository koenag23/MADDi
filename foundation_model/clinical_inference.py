# NOTE: NEED TO MOVE BACK INTO RTFM FOLDER TO MAKE IT WORK

'''
This notebook shows some example workflows of how to perform inference with TabuLa-8B.

For best performance, this notebook should be run with access to a GPU.

TabuLa-8B supports inference on zero- and few-shot tabular data (with the number of shots only limited by the context window of the model) and both categorical and continuous inputs. Below, we show examples of both.

TabuLa's inference uses pandas DataFrames to construct examples for downstream inference. We directly construct Pandas DataFrames below, but you can also read DataFrames from CSV files or any other source that can be converted to DataFrame.

Note about evaluation with labeled data: If you only want to perform efficient evaluation on data that is already labeled (i.e. to assess the accuracy of TabuLa on your own dataset), we provide separate code to do this which is likely to be more performant than the code in this notebook (which is optimized for simplicity/usability, not performance). Please see the README in the main repo for instructions on how to prepare your data for evaluation with our eval pipeline. Note that that eval pipeline (not the code in this notebook) is also what was used to evaluate TabuLa-8B on our paper.

Model loading and setup
First, load the model and tokenizer. It is important to use the TabuLa tokenizer (not the base Llama 3 tokenizer) due to the special tokens used for serialization.
'''

import pandas as pd
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig
import sys
sys.path.append('/Users/koenagupta/MADDi-1/foundation_model/rtfm/rtfm')

from rtfm.configs import TrainConfig, TokenizerConfig
from rtfm.inference_utils import InferenceModel
from rtfm.serialization.serializers import get_serializer
from rtfm.tokenization.text import prepare_tokenizer

train_config = TrainConfig(model_name="mlfoundations/tabula-8b", context_length=8192)

# If using a base llama model (not fine-tuned TabuLa),
# make sure to set add_serializer_tokens=False
# (because we do not want to use special tokens for 
# the base model which is not trained on them).
tokenizer_config = TokenizerConfig()

# Load the configuration
config = AutoConfig.from_pretrained(train_config.model_name)

# Set the torch_dtype to bfloat16 which matches TabuLa train/eval setup
config.torch_dtype = 'bfloat16'

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LlamaForCausalLM.from_pretrained(
    train_config.model_name, device_map="auto", config=config).to(device)

tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
serializer = get_serializer(train_config.serializer_cls)

tokenizer, model = prepare_tokenizer(
    model,
    tokenizer=tokenizer,
    pretrained_model_name_or_path=train_config.model_name,
    model_max_length=train_config.context_length,
    use_fast_tokenizer=tokenizer_config.use_fast_tokenizer,
    serializer_tokens_embed_fn=tokenizer_config.serializer_tokens_embed_fn,
    serializer_tokens=serializer.special_tokens
    if tokenizer_config.add_serializer_tokens
    else None,
)

inference_model = InferenceModel(model=model, tokenizer=tokenizer, serializer=serializer)

'''
Below is an example.

labeled_examples = pd.DataFrame(
    [
        {"location": "New York", "temperature": 22, "humidity": 65, "wind_speed": 12, "pressure": 1012, "month": "July",
         "weather_yesterday": "Sunny", "precipitation": 0, "visibility": 10, "weather_today": "Partly Sunny"},
        {"location": "Los Angeles", "temperature": 26, "humidity": 60, "wind_speed": 7, "pressure": 1015,
         "month": "July", "weather_yesterday": "Partly Sunny", "precipitation": 0, "visibility": 10, "weather_today": "Sunny"},
        {"location": "Chicago", "temperature": 18, "humidity": 70, "wind_speed": 15, "pressure": 1008, "month": "July",
         "weather_yesterday": "Partly Cloudy", "precipitation": 0.1, "visibility": 8, "weather_today": "Cloudy"},
        {"location": "Houston", "temperature": 30, "humidity": 80, "wind_speed": 10, "pressure": 1010, "month": "July",
         "weather_yesterday": "Rain", "precipitation": 0.5, "visibility": 7, "weather_today": "Rain"},
        {"location": "Phoenix", "temperature": 35, "humidity": 20, "wind_speed": 5, "pressure": 1005, "month": "July",
         "weather_yesterday": "Sunny", "precipitation": 0, "visibility": 10, "weather_today": "Sunny"},
        {"location": "Philadelphia", "temperature": 24, "humidity": 75, "wind_speed": 14, "pressure": 1009,
         "month": "July", "weather_yesterday": "Partly Cloudy", "precipitation": 0.2, "visibility": 9,
         "weather_today": "Partly Cloudy"},
        {"location": "San Antonio", "temperature": 28, "humidity": 68, "wind_speed": 11, "pressure": 1011,
         "month": "July", "weather_yesterday": "Rain", "precipitation": 0.4, "visibility": 8, "weather_today": "Rain"},
        {"location": "San Diego", "temperature": 22, "humidity": 65, "wind_speed": 10, "pressure": 1014,
         "month": "July", "weather_yesterday": "Sunny", "precipitation": 0, "visibility": 10, "weather_today": "Partly Sunny"},
        {"location": "Dallas", "temperature": 27, "humidity": 72, "wind_speed": 9, "pressure": 1007, "month": "July",
         "weather_yesterday": "Partly Cloudy", "precipitation": 0.3, "visibility": 9, "weather_today": "Cloudy"},
    ]
)
target_example = pd.DataFrame(
    [
        {"location": "San Jose", "temperature": 23, "humidity": 55, "wind_speed": 8, "pressure": 1013, "month": "July",
         "weather_yesterday": "Sunny", "precipitation": 0, "visibility": 10, "weather_today": "Sunny"},
    ]
)

output = inference_model.predict(
    target_example=target_example,
    target_colname="weather_today",
    target_choices=["Sunny", "Partly Sunny", "Cloudy", "Partly Cloudy", "Rain"],
    labeled_examples=labeled_examples,
)
print(f"Prediction for sample \n {target_example} \n is: {output}")

'''
file_path = '../preprocess_clinical/clinical.csv' # INSERT CSV FILE
df = pd.read_csv(file_path)
print(df)

# output = inference_model.predict(
#     target_example=target_example,
#     target_colname="weather_today",
#     target_choices=["Sunny", "Partly Sunny", "Cloudy", "Partly Cloudy", "Rain"],
#     labeled_examples=labeled_examples,
# )
# print(f"Prediction for sample \n {target_example} \n is: {output}")

'''
Prediction with continuous targets
TabuLa-8B is also trained to perform prediction on continuous targets. This is handled by bucketing the continuous inputs and treating these as categorical labels. Besides this bucketization step, the procedure is otherwise identical.

In order to ensure best performance, it is important to ensure that your serialization of the continuous targets matches exactly the expected format used during TabuLa's training.

Note that here we'll process the examples together to ensure the discretization is applied correctly, before splitting into train/target samples.
'''

# from rtfm.serialization.serialization_utils import discretize_continuous_column

# examples = pd.DataFrame(
#     [
#     {"location": "New York", "size_sqft": 1200, "bedrooms": 3, "bathrooms": 2, "age": 10, "lot_size_acres": 0.15, "garage": True, "price": 850},
#     {"location": "Los Angeles", "size_sqft": 1500, "bedrooms": 4, "bathrooms": 3, "age": 8, "lot_size_acres": 0.25, "garage": True, "price": 950},
#     {"location": "Chicago", "size_sqft": 1300, "bedrooms": 3, "bathrooms": 2, "age": 15, "lot_size_acres": 0.2, "garage": False, "price": 700},
#     {"location": "Houston", "size_sqft": 1700, "bedrooms": 4, "bathrooms": 3, "age": 5, "lot_size_acres": 0.3, "garage": True, "price": 650},
#     {"location": "Phoenix", "size_sqft": 1600, "bedrooms": 3, "bathrooms": 2, "age": 7, "lot_size_acres": 0.25, "garage": True, "price": 750},
#     {"location": "Philadelphia", "size_sqft": 1400, "bedrooms": 3, "bathrooms": 2, "age": 12, "lot_size_acres": 0.18, "garage": False, "price": 600},
#     {"location": "San Antonio", "size_sqft": 1800, "bedrooms": 4, "bathrooms": 3, "age": 3, "lot_size_acres": 0.4, "garage": True, "price": 700},
#     {"location": "San Diego", "size_sqft": 1550, "bedrooms": 3, "bathrooms": 2, "age": 9, "lot_size_acres": 0.22, "garage": True, "price": 850},
#     {"location": "Dallas", "size_sqft": 1450, "bedrooms": 3, "bathrooms": 2, "age": 11, "lot_size_acres": 0.19, "garage": True, "price": 700},
#     {"location": "San Jose", "size_sqft": 1600, "bedrooms": 4, "bathrooms": 3, "age": 6, "lot_size_acres": 0.2, "garage": False, "price": 800},
#     {"location": "Seattle", "size_sqft": 1800, "bedrooms": 4, "bathrooms": 2, "age": 10, "lot_size_acres": 0.2, "garage": False, "price": 925},
# ]
# )

# examples["price"] = discretize_continuous_column(examples["price"], num_buckets=4)
# target_choices = examples["price"].unique().tolist()

# target_example = examples.iloc[[0]]
# labeled_examples = examples.iloc[1:]


# output = inference_model.predict(
#     target_example=target_example,
#     target_colname="price",
#     target_choices=target_choices,
#     labeled_examples=labeled_examples,
# )
# print(f"Prediction for sample \n {target_example} \n is: {output}")
 