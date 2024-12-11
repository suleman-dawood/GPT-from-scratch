import os

folder_path = "Nietzsche"
all_text = ""  # Initialize an empty string to store combined text

# Loop through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            all_text += file.read()  # Append the content of each file