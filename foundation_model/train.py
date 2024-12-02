# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mlfoundations/tabula-8b")
model = AutoModelForCausalLM.from_pretrained("mlfoundations/tabula-8b")