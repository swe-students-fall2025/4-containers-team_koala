const ML_API_URL = "http://localhost:8080/predict";
const MIN_CONFIDENCE = 0.6;
const PREDICT_EVERY_N_FRAMES = 40;

const videoEl = document.getElementById("video");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl ? canvasEl.getContext("2d") : null;

const letterEl = document.getElementById("letter-display");
const confidenceEl = document.getElementById("confidence-display");

let frameCount = 0;
let inFlight = false;

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoEl.srcObject = stream;
    await videoEl.play();
  } catch (err) {
    console.error("Camera error:", err);
  }
}

// MediaPipe Setup
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6,
});

// drawing_utils globals:
const { drawConnectors, drawLandmarks } = window;

// MediaPipe Results
hands.onResults(async (results) => {
  if (!canvasEl || !ctx) return;

  // Resize canvas to match video
  if (videoEl.videoWidth && videoEl.videoHeight) {
    if (canvasEl.width !== videoEl.videoWidth || canvasEl.height !== videoEl.videoHeight) {
      canvasEl.width = videoEl.videoWidth;
      canvasEl.height = videoEl.videoHeight;
    }
  }

  // Clear and draw video frame
  ctx.save();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);
  ctx.restore();

  const multiHandLandmarks = results.multiHandLandmarks;
  if (!multiHandLandmarks || multiHandLandmarks.length === 0) return;

  const landmarks = multiHandLandmarks[0];

  // Draw landmarks
  drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 2 });
  drawLandmarks(ctx, landmarks, { color: "#FF0000", lineWidth: 1 });

  // Throttle ML API calls
  frameCount++;
  if (frameCount % PREDICT_EVERY_N_FRAMES !== 0 || inFlight) return;

  const points = landmarks.map((lm) => [lm.x, lm.y, lm.z]);

  inFlight = true;
  try {
    const res = await fetch(ML_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points }),
    });

    if (!res.ok) {
      console.error("ML API error:", res.status);
      return;
    }

    const data = await res.json(); // { letter, confidence }
    if (data && typeof data.letter === "string") {
      if (letterEl) letterEl.innerText = data.letter;
      if (confidenceEl && typeof data.confidence === "number") {
        confidenceEl.innerText = data.confidence.toFixed(2);
      }

      // Hook for assessment.html
      if (window.onPrediction) {
        window.onPrediction(data.letter, data.confidence, points);
      }
    }
  } catch (err) {
    console.error("Prediction error:", err);
  } finally {
    inFlight = false;
  }
});

// Feed video frames into MediaPipe
async function loop() {
  if (videoEl.readyState >= 2) {
    await hands.send({ image: videoEl });
  }
  requestAnimationFrame(loop);
}

startCamera().then(() => loop());
