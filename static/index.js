// thanks to
// - http://www.williammalone.com/articles/create-html5-canvas-javascript-drawing-app/
// - https://stackoverflow.com/questions/17656292/html5-canvas-support-in-mobile-phone-browser

let context
let paint
let clickX = []
let clickY = []
let clickDrag = []



function startCanvas() {
  let canvas = document.getElementById('canvas')
  context = canvas.getContext("2d")

  context.strokeStyle = "#000"
  context.lineJoin = "round"
  context.lineWidth = 10

  canvas.addEventListener("touchstart", function (e) {
    let touch = e.touches[0]
    let mouseEvent = new MouseEvent("mousedown", {
      clientX: touch.clientX,
      clientY: touch.clientY
    })
    canvas.dispatchEvent(mouseEvent)
  }, false)

  canvas.addEventListener("touchmove", function (e) {
    let touch = e.touches[0]
    let mouseEvent = new MouseEvent("mousemove", {
      clientX: touch.clientX,
      clientY: touch.clientY
    })
    canvas.dispatchEvent(mouseEvent)
  }, false)

  canvas.addEventListener("touchend", function (e) {
    let mouseEvent = new MouseEvent("mouseup")
    canvas.dispatchEvent(mouseEvent)
  }, false)

  $('#canvas').mousedown(function (e) {
    paint = true
    addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, false)
    redraw()
  })

  $('#canvas').mousemove(function (e) {
    if (paint) {
      addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true)
      redraw()
    }
  })

  $('#canvas').mouseup(function (e) {
    paint = false
    redraw()
  })

  $('#canvas').mouseleave(function (e) {
    paint = false
  })
}

function addClick(x, y, dragging) {
  clickX.push(x)
  clickY.push(y)
  clickDrag.push(dragging)
}

function clearCanvas() {
  context.clearRect(0, 0, 200, 200)
}

function resetCanvas() {
  clickX = []
  clickY = []
  clickDrag = []
  clearCanvas()
}

function redraw() {
  clearCanvas();
  context.lineJoin = "round";
  context.lineWidth = 10;  // <- MAKE THIS BIGGER
  context.strokeStyle = "#000";  // ensure it's black

  for (let i = 0; i < clickX.length; i++) {
    context.beginPath();
    if (clickDrag[i] && i) {
      context.moveTo(clickX[i - 1], clickY[i - 1]);
    } else {
      context.moveTo(clickX[i] - 1, clickY[i]);
    }
    context.lineTo(clickX[i], clickY[i]);
    context.closePath();
    context.stroke();
  }
}

function sendTrainingData() {
  pixels = getPixels()
  document.getElementById('pixels').value = pixels
  document.getElementById('train-form').submit()
}

function sendQuizData() {
  pixels = getPixels()
  document.getElementById('pixels').value = pixels
  document.getElementById('quiz-form').submit()
}

function sendPracticeData() {
  pixels = getPixels()
  document.getElementById('pixels').value = pixels
  document.getElementById('practice-form').submit()
}

function getPixels() {
  let tempCanvas = document.createElement('canvas')
  tempCanvas.width = 50
  tempCanvas.height = 50
  let tempCtx = tempCanvas.getContext('2d')

  // Draw scaled-down image
  tempCtx.drawImage(canvas, 0, 0, 50, 50)

  let rawPixels = tempCtx.getImageData(0, 0, 50, 50).data
  let pixels = []

  for (let i = 0; i < rawPixels.length; i += 4) {
    const r = rawPixels[i]
    const g = rawPixels[i + 1]
    const b = rawPixels[i + 2]

    // Convert RGB to grayscale using luminosity method
    const grayscale = 0.299 * r + 0.587 * g + 0.114 * b

    pixels.push(grayscale)
  }

  return pixels
}