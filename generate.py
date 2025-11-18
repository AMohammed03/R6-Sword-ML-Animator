import torch
import json
import numpy as np
import sys
import os
from model import AnimationGenerator, convert_flat_to_nested, KEYFRAME_TIMES

GUI_AVAILABLE = True
missing_modules = []

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
except ImportError as e:
    GUI_AVAILABLE = False
    missing_modules.append(f"tkinter: {e}")

try:
    import pyperclip
except ImportError as e:
    GUI_AVAILABLE = False
    missing_modules.append(f"pyperclip: {e}")
    GUI_AVAILABLE = False

def get_resource_pack(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def load_trained_model(model_path='trained_model.pth'):

    try:
        full_path = get_resource_pack(model_path)
        checkpoint = torch.load(full_path)
    except FileNotFoundError:
        sys.exit(1)

    model = AnimationGenerator(
        input_size=2,
        output_size=checkpoint['num_features'] * checkpoint['max_keyframes'],
        hidden_sizes=checkpoint['config']['hidden_sizes'],
        dropout=checkpoint['config']['dropout_rate']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint

def generate_new_animation(pattern, output_file='generated_animation.json'):
    model, checkpoint = load_trained_model()

    with torch.no_grad():
        pattern_tensor = torch.FloatTensor([pattern])
        prediction = model(pattern_tensor).numpy()[0]

        keyframes = prediction.reshape(
            checkpoint['max_keyframes'],
            checkpoint['num_features']
        )

        time_column = np.array(KEYFRAME_TIMES).reshape(-1, 1)
        keyframes_with_time = np.concatenate([time_column, keyframes], axis=1)

        nested_keyframes = [
            convert_flat_to_nested(kf, checkpoint['feature_cols'])
            for kf in keyframes_with_time
        ]

        animation = {
            'pattern': f"{pattern[0]}-{pattern[1]}",
            'keyframes': nested_keyframes
        }

        with open(output_file, 'w') as f:
            json.dump(animation, f, indent=2)

        return animation

def get_user_input():

    while True:
        try:
            start = input("Enter start position (1-9): ")
            end = input("Enter end position (1-9): ")

            start_num = int(start)
            end_num = int(end)

            if start_num < 1 or start_num > 9 or end_num < 1 or end_num > 9:
                print("Error: Positions must be between 1 and 9")
                continue

            return [start_num, end_num]
        
        except ValueError:
            print("Error: Please enter valid numbers. Try again.")

        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            sys.exit(0)

class AnimationGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("R6 Sword Animation Generator")
        self.root.geometry("400x500")
        self.root.resizable(False, False)


        self.model = None
        self.checkpoint = None
        self.animation_data = None
        self.load_model()

        self.create_widget()

    def load_model(self):
        try:
            model_path = get_resource_pack('trained_model.pth')
            self.checkpoint = torch.load(model_path)
            self.model = AnimationGenerator(
                input_size=2,
                output_size=self.checkpoint['num_features'] * self.checkpoint['max_keyframes'],
                hidden_sizes=self.checkpoint['config']['hidden_sizes'],
                dropout=self.checkpoint['config']['dropout_rate']
            )
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.model.eval()
        except FileNotFoundError:
            messagebox.showerror("Error", "Could not find 'trained_model.pth'\nMake sure it's in the same directory.")
            self.root.destroy()

    def create_widget(self):
        title_label = tk.Label(
            self.root,
            text="R6 Sword Animation Generator",
            font=("Arial", 16, "bold"),
            pady=10
        )
        title_label.pack()

        input_frame = tk.LabelFrame(self.root, text="Slash Pattern", padx=20, pady=20)
        input_frame.pack(padx=20, pady=10, fill="x")

        numpad_frame = tk.Frame(input_frame)
        numpad_frame.pack(pady=10)

        tk.Label(numpad_frame, text="Pattern positions (numpad layout):", font=("Arial", 10)).pack()\
        
        numpad_display = tk.Label(
            numpad_frame,
            text="1 2 3\n4 5 6\n7 8 9",
            font=("Courier", 14, "bold"),
            fg="blue"
        )
        numpad_display.pack(pady=5)

        input_fields_frame = tk.Frame(input_frame)
        input_fields_frame.pack(pady=10)

        tk.Label(input_fields_frame, text="Start Position:", font=("Arial", 10)).grid(row=0, column=0, padx=5, sticky="e")
        self.start_var = tk.StringVar()
        start_entry = tk.Entry(input_fields_frame, textvariable=self.start_var, width=5, font=("Arial", 12))
        start_entry.grid(row=0, column=1, padx=5)

        tk.Label(input_fields_frame, text="→", font=("Arial", 14)).grid(row=0, column=2, padx=10)

        tk.Label(input_fields_frame, text="End Position:", font=("Arial", 10)).grid(row=0, column=2, padx=10)
        self.end_var = tk.StringVar()
        end_entry = tk.Entry(input_fields_frame, textvariable=self.end_var, width=5, font=("Arial", 12))
        end_entry.grid(row=0, column=4, padx=5)


        generate_btn = tk.Button(
            input_frame,
            text="Generate Button",
            command=self.generate_animation,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10,
            cursor="hand2"
        )
        generate_btn.pack(pady=10)

        output_frame = tk.LabelFrame(self.root, text="Generated JSON", padx=10, pady=10)
        output_frame.pack(padx=20, pady=10, fill="both", expand=True)

        self.json_output = scrolledtext.ScrolledText(
            output_frame,
            width=60,
            height=20,
            font=("Courier", 9),
            wrap=tk.WORD
        )
        self.json_output.pack(fill="both", expand=True)

        button_frame = tk.Frame(output_frame)
        button_frame.pack(pady=10)

        copy_btn = tk.Button(
            button_frame,
            text="Copy to clipboard",
            command=self.copy_to_clipboard,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=5,
            cursor="hand2"
        )
        copy_btn.pack(side="left", padx=5)

        save_btn = tk.Button(
            button_frame,
            text="Save to File",
            command=self.save_to_file,
            bg="#FF9800",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=5,
            cursor="hand2"
        )
        save_btn.pack(side="left", padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_var = tk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor="w",
            font=("Arial", 9)
        )
        status_var.pack(side="bottom", fill="x")

    def generate_animation(self):
        try:
            start = int(self.start_var.get())
            end = int(self.end_var.get())
            
            if start < 1 or start > 9 or end < 1 or end > 9:
                messagebox.showerror("Invalid Input", "Positions must be between 1 and 9")
                return
            
            self.status_var.set(f"Generating animation for pattern {start}-{end}...")
            self.root.update()
            
            pattern = [start, end]
            
            with torch.no_grad():
                pattern_tensor = torch.FloatTensor([pattern])
                prediction = self.model(pattern_tensor).numpy()[0]
                
                keyframes = prediction.reshape(
                    self.checkpoint['max_keyframes'],
                    self.checkpoint['num_features']
                )
                
                time_column = np.array(KEYFRAME_TIMES).reshape(-1, 1)
                keyframes_with_time = np.concatenate([time_column, keyframes], axis=1)
                
                nested_keyframes = [
                    convert_flat_to_nested(kf, self.checkpoint['feature_cols'])
                    for kf in keyframes_with_time
                ]
                
                self.animation_data = {
                    'pattern': f"{start}-{end}",
                    'keyframes': nested_keyframes
                }
            
            json_str = json.dumps(self.animation_data, indent=2)
            self.json_output.delete(1.0, tk.END)
            self.json_output.insert(1.0, json_str)
            
            self.status_var.set(f"Animation generated successfully! Pattern: {start}-{end}")
            messagebox.showinfo("Success", f"Animation for pattern {start}-{end} generated!\n\nClick 'Copy to Clipboard' to use in Roblox Studio.")
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers (1-9)")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.status_var.set("Error occurred")

    def copy_to_clipboard(self):
        """Copy JSON to clipboard."""
        json_text = self.json_output.get(1.0, tk.END).strip()
        if not json_text:
            messagebox.showwarning("No Data", "Generate an animation first!")
        return

        try:
            pyperclip.copy(json_text)
            self.status_var.set("Copied to clipboard!")
            messagebox.showinfo("Copied", "JSON copied to clipboard!\n\nPaste it into your Roblox AnimationImporter script.")
        except:
            self.root.clipboard_clear()
            self.root.clipboard_append(json_text)
            self.status_var.set("Copied to clipboard!")
            messagebox.showinfo("Copied", "JSON copied to clipboard!\n\nPaste it into your Roblox AnimationImporter script.")
 
    def save_to_file(self):
        json_text = self.json_output.get(1.0, tk.END).strip()
        if not json_text:
            messagebox.showwarning("No Data", "Generate an animation first!")
            return
        
        try:
            pattern = self.animation_data['pattern'].replace('-', '_')
            filename = f"animation_{pattern}.json"
            
            with open(filename, 'w') as f:
                f.write(json_text)
            
            self.status_var.set(f"✓ Saved to {filename}")
            messagebox.showinfo("Saved", f"Animation saved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file:\n{str(e)}")

def main_cli():
    if len(sys.argv) == 3:
        try:
            pattern = [int(sys.argv[1]), int(sys.argv[2])]
            if pattern[0] < 1 or pattern[0] > 9 or pattern[1] < 1 or pattern[1] > 9:
                print("Error: Pattern positions must be between 1 and 9")
                sys.exit(1)
        except ValueError:
            print("Error: Invalid pattern. Usage: python generate.py <start> <end>")
            print("Example: python generate.py 6 4")
            sys.exit(1)
    else:
        pattern = get_user_input()
    
    output_file = f"animation_{pattern[0]}_{pattern[1]}.json"
    generate_new_animation(pattern, output_file)
    
    print("\n" + "=" * 50)
    another = input("Generate another animation? (y/n): ").strip().lower()
    if another == 'y' or another == 'yes':
        print()
        main_cli()
    else:
        print("\nThank you for using Sword Animation Generator!")

def main_gui():
    """GUI mode."""
    if not GUI_AVAILABLE:
        print("Error: GUI libraries not available.")
        print("Missing modules:")
        for mod in missing_modules:
            print(f"  - {mod}")
        print("\nTry running in CLI mode: python generate.py --cli")
        sys.exit(1)
    
    root = tk.Tk()
    app = AnimationGeneratorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    # Check if running in GUI mode (default) or CLI mode
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        # CLI mode: python generate.py --cli
        sys.argv.pop(1)  # Remove --cli flag
        main_cli()
    elif len(sys.argv) > 1:
        # CLI with arguments: python generate.py 6 4
        main_cli()
    else:
        # Default: GUI mode
        main_gui()