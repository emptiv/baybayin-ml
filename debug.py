import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("scripts/baybayin_model_v2.keras")  # Update as needed

class DrawApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Baybayin Classifier")

        self.canvas_size = 28  # draw on 28x28
        self.upscaled_size = 50  # model input size

        self.display_scale = 10  # upscale for visibility
        self.canvas = tk.Canvas(master, width=self.canvas_size * self.display_scale,
                                height=self.canvas_size * self.display_scale, bg='white')
        self.canvas.pack()

        self.button_frame = tk.Frame(master)
        self.button_frame.pack()

        tk.Button(self.button_frame, text="Predict", command=self.predict).pack(side="left")
        tk.Button(self.button_frame, text="Clear", command=self.clear).pack(side="left")

        self.label = tk.Label(master, text="Draw a character and click Predict")
        self.label.pack()

        self.image1 = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image1)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        # Scale down mouse coordinates to 28x28 space
        x = event.x // self.display_scale
        y = event.y // self.display_scale
        r = 0.4  # square brush half-size

        # Draw square on image
        self.draw.rectangle([x - r, y - r, x + r, y + r], fill=0)

        # Draw square on screen canvas
        self.canvas.create_rectangle(
            (x - r) * self.display_scale, (y - r) * self.display_scale,
            (x + r + 1) * self.display_scale, (y + r + 1) * self.display_scale,
            fill='black', outline='black'
        )

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=255)
        self.label.config(text="Draw a character and click Predict")

    def predict(self):
        # Resize the 28x28 image to 50x50
        img = self.image1.resize((50, 50))
        img = ImageOps.invert(img)
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # shape: (1, 50, 50, 1)

        preds = model.predict(img_array)
        label = np.argmax(preds)

        corrected_decoder = {0: 'e_i', 1: 'o_u', 2: 'a'}
        self.label.config(text=f"Prediction: {corrected_decoder[label]}")

# Run app
root = tk.Tk()
app = DrawApp(root)
root.mainloop()
