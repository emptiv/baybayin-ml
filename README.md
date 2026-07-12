# baybayin character recognition app

an interactive desktop app built with python and tkinter that recognizes handwritten baybayin script characters using [convolutional neural network (cnn) models](https://github.com/emptiv/baybayin-ml-archive). you can draw characters directly on a digital canvas and get real-time model predictions based on specific lessons.

---

### important notes
- this project is experimental and nowhere near perfect. my models may seem to have 100% confidence, but expect variance in accuracy!
- the loaded model can *only* predict characters belonging to its corresponding lesson. if you are on **lesson 1**, it won't recognize characters from lesson 2 or 3.
- known issue: the character **"PA"** has some recognition quirks and might not predict correctly.

---

## prerequisites
- **python 3.12** (strict requirement for tensorflow compatibility)

---

## installation & setup

1. **clone the project** and ensure your structure looks like this:
   ```text
   ├── main.py
   └── models/ (contains lesson_1.keras, etc.)
   ```

2. **install packages:**
   ```bash
   pip install pillow numpy tensorflow
   ```

3. **run the app:**
   ```bash
   python main.py
   ```

---

## how to use
- **select a lesson** from the dropdown to load its specific characters.
- **draw** on the canvas with your left mouse button.
- click **predict** to see the result, or **clear** to reset the canvas.