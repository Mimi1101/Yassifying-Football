from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("MLBtrio/genz-slang-dataset", split="train")

# Open a text file to write the slang and descriptions
with open("genz_slang.txt", "w", encoding="utf-8") as file:
    for entry in dataset:
        slang = entry.get("Slang", "").strip()
        description = entry.get("Description", "").strip()
        if slang and description:
            file.write(f"{slang}: {description}\n")
