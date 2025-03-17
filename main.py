import tkinter as tk
from tkinter import scrolledtext
import cv2
from PIL import Image, ImageTk
import threading
import time

from utils import object_detection, language_model, prompt_processing, video_utils

class VLMApp:
    def __init__(self, window, window_title="VLM with Live Feed"):
        self.window = window
        self.window.title(window_title)

        self.camera = video_utils.open_camera()
        if not self.camera.isOpened():
            print("Error: Could not open camera.")
            exit()

        self.tokenizer, self.model = language_model.load_model()

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.prompt_label = tk.Label(window, text="Enter Prompt:")
        self.prompt_label.pack()

        self.prompt_entry = tk.Entry(window, width=50)
        self.prompt_entry.pack()

        self.process_button = tk.Button(window, text="Process", command=self.process_prompt)
        self.process_button.pack()
        
        self.quit_button = tk.Button(window, text="Quit", command=self.close) #Added quit button
        self.quit_button.pack()

        self.response_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=10)
        self.response_text.pack()

        self.update()

        self.prompt_entry.focus_set() #force focus
        self.window.mainloop()

    def update(self):
        ret, frame = self.camera.read()
        if ret:
            detections = object_detection.detect_objects(frame)
            video_utils.display_frame(frame, detections)  # Draw boxes on frame.

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(10, self.update)  # Update every 10 milliseconds

    def process_prompt(self, event=None):
        try:
            print("process_prompt function called")
            user_prompt = self.prompt_entry.get()
            if user_prompt:
                print(f"User prompt: {user_prompt}")
                detections = object_detection.detect_objects(video_utils.capture_frame(self.camera))
                prompt = prompt_processing.create_prompt(detections, user_prompt)
                print(f"Generated prompt: {prompt}")

                start_time = time.time()
                response = language_model.generate_text(prompt, self.tokenizer, self.model)
                end_time = time.time()
                llm_time = end_time - start_time
                print(f"LLM response time: {llm_time:.4f} seconds")

                # Insert the response into the scrolledtext widget
                self.response_text.insert(tk.END, f"Prompt: {user_prompt}\nResponse: {response}\n\n")

                self.prompt_entry.delete(0, tk.END)  # Clear the input box
            else:
                print("No prompt entered.")
        except Exception as e:
            print(f"Error in process_prompt: {e}")

    def close(self):
        video_utils.close_camera(self.camera)
        self.window.destroy()

if __name__ == "__main__":
    print("Main Start")
    window = tk.Tk()
    print("Window Finished")
    app = VLMApp(window)
    print("APP Finished")
    window.protocol("WM_DELETE_WINDOW", app.close)
    print(".protocol")