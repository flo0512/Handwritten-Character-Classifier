import tkinter as tk
from PIL import Image, ImageGrab, ImageOps
import numpy as np
import tensorflow as tf


LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]


model = tf.keras.models.load_model(
    r"C:\Users\grufl\Documents\GitHub\Handwritten-Character-Classifier\my_model.keras")


root = tk.Tk()
root.configure(bg="grey", width=400, height=400)
root.title("Handwritten Character Classifier")

canvas_width = 300
canvas_height = 300
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

drawing = False
last_x, last_y = None, None


def start_draw(event):
    global drawing, last_x, last_y
    drawing = True
    last_x, last_y = event.x, event.y


def draw(event):
    global last_x, last_y
    if drawing:
        canvas.create_line(last_x, last_y, event.x, event.y,
                           fill='black', width=15, capstyle=tk.ROUND, smooth=True)
        last_x, last_y = event.x, event.y


def stop_draw(event):
    global drawing
    drawing = False


def clear_canvas():
    canvas.delete("all")
    prediction_label.config(text="")


def predict():

    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    img = ImageOps.fit(img, (28, 28), method=Image.BICUBIC,
                       centering=(0.5, 0.5))

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    pred = model.predict(img_array)
    predicted_class = np.argmax(pred)
    predicted_char = LABELS[predicted_class]
    prediction_label.config(
        text=f"Prediction: {predicted_char}")


canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_draw)


btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="predict", command=predict).pack(
    side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="delete", command=clear_canvas).pack(
    side=tk.LEFT, padx=5)


prediction_label = tk.Label(root, text="", font=("Arial", 16))
prediction_label.pack(pady=10)

root.mainloop()
