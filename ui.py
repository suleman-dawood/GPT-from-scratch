import tkinter as tk
from tkinter import messagebox
from generate import *

# Function to call the generate function and display output
def func():
    try:
        # Call the generate function and capture its output
        result = generate_text(150)
        # Display the output in the text box
        output_text.delete(1.0, tk.END)  # Clear previous output
        output_text.insert(tk.END, result)
    except Exception as e:
        # Show an error message if an exception occurs
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

# Create the main application window
root = tk.Tk()
root.title("Generate Output UI")

# Create a button to trigger the generate function
run_button = tk.Button(root, text="Generate", command=generate_text, font=("Arial", 14), bg="blue", fg="white")
run_button.pack(pady=10)

# Create a text box to display the output
output_text = tk.Text(root, wrap=tk.WORD, height=20, width=60, font=("Arial", 12))
output_text.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()