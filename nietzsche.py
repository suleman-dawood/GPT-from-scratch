import os

folder_path = "Nietzsche"
all_text = ""

# looping through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            all_text += file.read()  # concatenating the contents

