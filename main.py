import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow import keras


# lesson information and models used
LESSONS = {
    1: {
        "model": "models/lesson_1.keras",
        "classes": ["A", "O/U", "E/I"]
    },
    2: {
        "model": "models/lesson_2.keras",
        "classes": ["PA", "KA", "NA"]
    },
    3: {
        "model": "models/lesson_3.keras",
        "classes": ["HA", "BA", "GA"]
    },
    4: {
        "model": "models/lesson_4.keras",
        "classes": ["SA", "DA/RA", "TA"]
    },
    5: {
        "model": "models/lesson_5.keras",
        "classes": ["NGA", "WA", "LA"]
    },
    6: {
        "model": "models/lesson_6.keras",
        "classes": ["MA", "YA"]
    }
}

# canvas settings
WIDTH = 300
HEIGHT = 300
BRUSH_SIZE = 8

# main window
root = tk.Tk()
root.title("Baybayin Character Recognition")
root.resizable(False, False)

# variables
selected_lesson = tk.IntVar(value=1)
current_model = None
current_classes = None


def load_selected_model(*args):
    global current_model, current_classes

    lesson = selected_lesson.get()

    info = LESSONS[lesson]

    current_model = keras.models.load_model(
        info["model"],
        compile=False
    )

    current_classes = info["classes"]

    lesson_label.config(
        text="Characters: " + "   ".join(current_classes)
    )

    result_label.config(text="")


def paint(event):
    x = event.x
    y = event.y
    r = BRUSH_SIZE

    canvas.create_oval(
        x-r,
        y-r,
        x+r,
        y+r,
        fill="black",
        outline="black"
    )

    draw.ellipse(
        (x-r, y-r, x+r, y+r),
        fill="black"
    )


def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, WIDTH, HEIGHT), fill="white")
    result_label.config(text="")


def predict():

    img = image.resize((50, 50))

    img = np.array(img).astype("float32")

    img = 255 - img

    img /= 255.0

    img = img.reshape(1, 50, 50, 1)

    prediction = current_model.predict(img, verbose=0)

    index = np.argmax(prediction)

    confidence = prediction[0][index]

    result_label.config(
        text=f"Prediction: {current_classes[index]}"
    )


# lesson selector
tk.Label(
    root,
    text="Select Lesson",
    font=("Arial", 12, "bold")
).pack(pady=(10, 0))

lesson_dropdown = ttk.Combobox(
    root,
    values=[1, 2, 3, 4, 5, 6],
    textvariable=selected_lesson,
    state="readonly",
    width=10
)

lesson_dropdown.pack()

lesson_dropdown.bind(
    "<<ComboboxSelected>>",
    load_selected_model
)

lesson_label = tk.Label(
    root,
    text="",
    font=("Arial", 11)
)

lesson_label.pack(pady=5)

# drawing canvas
canvas = tk.Canvas(
    root,
    width=WIDTH,
    height=HEIGHT,
    bg="white",
    cursor="cross"
)

canvas.pack(pady=5)

image = Image.new("L", (WIDTH, HEIGHT), "white")
draw = ImageDraw.Draw(image)

canvas.bind("<B1-Motion>", paint)

# buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

predict_button = tk.Button(
    button_frame,
    text="Predict",
    width=12,
    command=predict
)

predict_button.grid(row=0, column=0, padx=5)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    width=12,
    command=clear_canvas
)

clear_button.grid(row=0, column=1, padx=5)

# result
result_label = tk.Label(
    root,
    text="",
    font=("Arial", 13, "bold")
)

result_label.pack(pady=(5, 10))

# load default model
load_selected_model()

root.mainloop()