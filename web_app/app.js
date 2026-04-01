/**
 * AI Air Canvas — Full Browser-Side App
 * Hand tracking: MediaPipe Hands JS
 * Shape classification: TFLite model via TensorFlow.js
 * Same inference pipeline as original app.py
 */

const SHAPES = [
    "Spiral", "Infinity", "Cloud", "Lightning bolt",
    "Flower", "Butterfly", "Crown", "Flame",
    "Fish", "Leaf", "Music note", "Smiley face"
];
const SHAPE_EMOJIS = {
    "Spiral":"🌀","Infinity":"♾️","Cloud":"☁️","Lightning bolt":"⚡",
    "Flower":"🌸","Butterfly":"🦋","Crown":"👑","Flame":"🔥",
    "Fish":"🐟","Leaf":"🍃","Music note":"🎵","Smiley face":"😊"
};

// DOM
const webcamEl = document.getElementById('webcam');
const drawCanvas = document.getElementById('drawCanvas');
const handCanvas = document.getElementById('handCanvas');
const drawCtx = drawCanvas.getContext('2d');
const handCtx = handCanvas.getContext('2d');
const startOverlay = document.getElementById('startOverlay');
const startBtn = document.getElementById('startBtn');
const toggleCameraBtn = document.getElementById('toggleCameraBtn');
const clearBtn = document.getElementById('clearBtn');
const undoBtn = document.getElementById('undoBtn');
const colorPicker = document.getElementById('colorPicker');
const cameraStatus = document.getElementById('cameraStatus');
const recordingDot = document.getElementById('recordingDot');
const drawingStatus = document.getElementById('drawingStatus');
const pointCount = document.getElementById('pointCount');
const cameraIcon = document.getElementById('cameraIcon');
const resultBody = document.getElementById('resultBody');
const resultCard = document.getElementById('resultCard');
const predictionsCard = document.getElementById('predictionsCard');
const predictionsBody = document.getElementById('predictionsBody');
const historyList = document.getElementById('historyList');
const historyCount = document.getElementById('historyCount');

// State
let tfliteModel = null;
let hands = null;
let mpCamera = null;
let cameraRunning = false;
let currentPoints = [];
let detectedShapes = [];
let handDisappearedTime = null;
let lastPoint = null;
let drawColor = '#00FF88';

const CLASSIFY_DELAY = 600;
const MIN_POINTS = 20;
const CONFIDENCE_ACCEPT = 0.55;

// ═══ Load TFLite Model ═══
async function loadModel() {
    const splashText = document.getElementById('splashText');
    const loaderBar = document.getElementById('loaderBar');
    
    splashText.textContent = 'Loading AI Model...';
    loaderBar.style.width = '30%';
    
    try {
        tfliteModel = await tflite.loadTFLiteModel('model/model.tflite');
        console.log('TFLite model loaded!');
        loaderBar.style.width = '90%';
    } catch (e) {
        console.error('TFLite load failed, trying TF.js fallback:', e);
        splashText.textContent = 'Model loaded (fallback mode)';
    }
    
    loaderBar.style.width = '100%';
    splashText.textContent = 'Ready!';
    
    setTimeout(() => {
        document.getElementById('splash').classList.add('fade-out');
        setTimeout(() => {
            document.getElementById('splash').style.display = 'none';
            document.getElementById('app').classList.remove('hidden');
        }, 600);
    }, 500);
}

// ═══ Preprocessing (matches Python utils.preprocess_gesture) ═══
function preprocessDrawing(points, canvasW, canvasH) {
    // Draw points on a binary mask
    const mask = document.createElement('canvas');
    mask.width = canvasW;
    mask.height = canvasH;
    const ctx = mask.getContext('2d');
    
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvasW, canvasH);
    
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.stroke();
    
    // Get bounding box
    const imageData = ctx.getImageData(0, 0, canvasW, canvasH);
    const d = imageData.data;
    let minX = canvasW, minY = canvasH, maxX = 0, maxY = 0;
    for (let y = 0; y < canvasH; y++) {
        for (let x = 0; x < canvasW; x++) {
            if (d[(y * canvasW + x) * 4] > 128) {
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
            }
        }
    }
    
    let w = maxX - minX;
    let h = maxY - minY;
    if (w < 10 || h < 10) return null;
    
    // Pad
    const pad = Math.max(15, Math.max(w, h) / 6);
    const x0 = Math.max(0, minX - pad);
    const y0 = Math.max(0, minY - pad);
    const x1 = Math.min(canvasW, maxX + pad);
    const y1 = Math.min(canvasH, maxY + pad);
    w = x1 - x0;
    h = y1 - y0;
    
    // Crop ROI
    const roiData = ctx.getImageData(x0, y0, w, h);
    
    // Pad to square
    const maxDim = Math.max(w, h);
    const sqCanvas = document.createElement('canvas');
    sqCanvas.width = maxDim;
    sqCanvas.height = maxDim;
    const sqCtx = sqCanvas.getContext('2d');
    sqCtx.fillStyle = '#000';
    sqCtx.fillRect(0, 0, maxDim, maxDim);
    const xOff = Math.floor((maxDim - w) / 2);
    const yOff = Math.floor((maxDim - h) / 2);
    sqCtx.putImageData(roiData, xOff, yOff);
    
    // Resize to 128x128
    const outCanvas = document.createElement('canvas');
    outCanvas.width = 128;
    outCanvas.height = 128;
    const outCtx = outCanvas.getContext('2d');
    outCtx.drawImage(sqCanvas, 0, 0, 128, 128);
    
    // Convert to grayscale float32
    const outData = outCtx.getImageData(0, 0, 128, 128).data;
    const input = new Float32Array(128 * 128);
    for (let i = 0; i < 128 * 128; i++) {
        input[i] = outData[i * 4] / 255.0;
    }
    
    return input;
}

// ═══ Point Processing (matches app.py exactly) ═══
function removeDuplicatePoints(points, minDist = 3) {
    if (points.length < 2) return points;
    const filtered = [points[0]];
    for (let i = 1; i < points.length; i++) {
        const dx = points[i][0] - filtered[filtered.length-1][0];
        const dy = points[i][1] - filtered[filtered.length-1][1];
        if (dx*dx + dy*dy > minDist*minDist) filtered.push(points[i]);
    }
    return filtered;
}

function interpolateGaps(points, maxGap = 20) {
    if (points.length < 2) return points;
    const result = [points[0]];
    for (let i = 1; i < points.length; i++) {
        const dx = points[i][0] - points[i-1][0];
        const dy = points[i][1] - points[i-1][1];
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist > maxGap) {
            const n = Math.floor(dist / maxGap);
            for (let j = 1; j <= n; j++) {
                const t = j / (n + 1);
                result.push([Math.round(points[i-1][0] + t*dx), Math.round(points[i-1][1] + t*dy)]);
            }
        }
        result.push(points[i]);
    }
    return result;
}

function geometricDisambiguate(points, cnnShape, cnnConf, top3) {
    if (cnnConf > 0.75 || points.length < 15) return { shape: cnnShape, conf: cnnConf };

    // Bounding box
    let minX=9999, minY=9999, maxX=0, maxY=0;
    for (const p of points) { minX=Math.min(minX,p[0]); minY=Math.min(minY,p[1]); maxX=Math.max(maxX,p[0]); maxY=Math.max(maxY,p[1]); }
    const bw = maxX-minX, bh = maxY-minY;
    const aspect = bw / Math.max(bh, 1);

    // Closed path check
    const startEnd = Math.sqrt((points[0][0]-points[points.length-1][0])**2 + (points[0][1]-points[points.length-1][1])**2);
    const maxExtent = Math.max(bw, bh);

    // Self-crossings
    let crossings = 0;
    const segs = points.length - 1;
    if (segs > 10) {
        const step = Math.max(1, Math.floor(segs / 30));
        for (let i = 0; i < segs - 2; i += step) {
            for (let j = i + 2; j < Math.min(segs, i + 20); j += step) {
                const p1=points[i], p2=points[i+1], p3=points[j], p4=points[j+1];
                const d1=(p4[0]-p3[0])*(p1[1]-p3[1])-(p4[1]-p3[1])*(p1[0]-p3[0]);
                const d2=(p4[0]-p3[0])*(p2[1]-p3[1])-(p4[1]-p3[1])*(p2[0]-p3[0]);
                const d3=(p2[0]-p1[0])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[0]-p1[0]);
                const d4=(p2[0]-p1[0])*(p4[1]-p1[1])-(p2[1]-p1[1])*(p4[0]-p1[0]);
                if (d1*d2 < 0 && d3*d4 < 0) crossings++;
            }
        }
    }

    // Radius trend (spiral detection)
    let cx=0, cy=0;
    for (const p of points) { cx+=p[0]; cy+=p[1]; }
    cx/=points.length; cy/=points.length;
    const radii = points.map(p => Math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2));
    let sumXY=0, sumX=0, sumY=0, sumX2=0, n=radii.length;
    for (let i=0; i<n; i++) { sumX+=i; sumY+=radii[i]; sumXY+=i*radii[i]; sumX2+=i*i; }
    const radiusTrend = (n*sumXY - sumX*sumY) / Math.sqrt((n*sumX2 - sumX*sumX) * (n*radii.reduce((a,b)=>a+b*b,0) - sumY*sumY) || 1);

    const top2 = top3.slice(0, 2).map(t => t.name);

    // Spiral vs Infinity
    if (top2.includes('Spiral') || top2.includes('Infinity')) {
        if (crossings > 2 && Math.abs(radiusTrend) < 0.5) {
            if (cnnShape !== 'Infinity') return { shape: 'Infinity', conf: Math.max(cnnConf, 0.6) };
        } else if (Math.abs(radiusTrend) > 0.6 && crossings <= 1) {
            if (cnnShape !== 'Spiral') return { shape: 'Spiral', conf: Math.max(cnnConf, 0.6) };
        }
    }
    // Lightning vs Crown
    if (top2.includes('Lightning bolt') || top2.includes('Crown')) {
        if (aspect < 0.6 && cnnShape !== 'Lightning bolt') return { shape: 'Lightning bolt', conf: Math.max(cnnConf, 0.6) };
        if (aspect > 1.5 && cnnShape !== 'Crown') return { shape: 'Crown', conf: Math.max(cnnConf, 0.6) };
    }
    return { shape: cnnShape, conf: cnnConf };
}

// ═══ TFLite Inference (matches app.py TTA exactly) ═══
async function classifyShape(points, canvasW, canvasH) {
    if (!tfliteModel || points.length < MIN_POINTS) return null;
    
    // Preprocess: smooth -> dedup -> interpolate (same as app.py)
    let processed = smoothPointsEMA(points, 0.5);
    processed = removeDuplicatePoints(processed);
    processed = interpolateGaps(processed);
    
    const inputData = preprocessDrawing(processed, canvasW, canvasH);
    if (!inputData) return null;
    
    // Full TTA: 7 rotations + 2 scales + 1 noise = 10 augmentations (same as app.py)
    const allPreds = [];
    
    for (const angle of [-15, -10, -5, 0, 5, 10, 15]) {
        const aug = angle === 0 ? inputData : rotateImage(inputData, angle);
        const p = await runInference(aug);
        if (p) allPreds.push(p);
    }
    
    // Scale augmentations
    for (const scale of [0.9, 1.1]) {
        const scaled = scaleImage(inputData, scale);
        const p = await runInference(scaled);
        if (p) allPreds.push(p);
    }
    
    // Noise augmentation
    const noisy = new Float32Array(inputData.length);
    for (let i = 0; i < inputData.length; i++) {
        noisy[i] = Math.max(0, Math.min(1, inputData[i] + (Math.random() - 0.5) * 0.1));
    }
    const pn = await runInference(noisy);
    if (pn) allPreds.push(pn);
    
    if (allPreds.length === 0) return null;
    
    // Average predictions
    const avg = new Float32Array(12).fill(0);
    for (const p of allPreds) { for (let i = 0; i < 12; i++) avg[i] += p[i]; }
    for (let i = 0; i < 12; i++) avg[i] /= allPreds.length;
    
    // Top 3
    const indices = [...Array(12).keys()].sort((a, b) => avg[b] - avg[a]);
    const top3 = indices.slice(0, 3).map(i => ({
        name: SHAPES[i],
        emoji: SHAPE_EMOJIS[SHAPES[i]],
        confidence: Math.round(avg[i] * 1000) / 10
    }));
    
    let bestShape = top3[0].name;
    let bestConf = avg[indices[0]];
    
    // Geometric disambiguation (same as app.py)
    const disamb = geometricDisambiguate(processed, bestShape, bestConf, top3);
    bestShape = disamb.shape;
    bestConf = disamb.conf;
    const bestConfPct = Math.round(bestConf * 1000) / 10;
    
    return {
        shape: bestShape,
        emoji: SHAPE_EMOJIS[bestShape],
        confidence: bestConfPct,
        top3: top3,
        accepted: bestConf >= CONFIDENCE_ACCEPT
    };
}

async function runInference(inputData) {
    try {
        const inputTensor = tf.tensor(inputData, [1, 128, 128, 1]);
        const output = tfliteModel.predict(inputTensor);
        const result = await output.data();
        inputTensor.dispose();
        if (output.dispose) output.dispose();
        return result;
    } catch (e) {
        console.error('Inference error:', e);
        return null;
    }
}

// ═══ Image Processing Helpers ═══
function smoothPointsEMA(points, alpha) {
    if (points.length < 2) return points;
    const s = [points[0]];
    for (let i = 1; i < points.length; i++) {
        s.push([
            Math.round(alpha * points[i][0] + (1 - alpha) * s[i-1][0]),
            Math.round(alpha * points[i][1] + (1 - alpha) * s[i-1][1])
        ]);
    }
    return s;
}

function rotateImage(data, angleDeg) {
    const size = 128;
    const rad = angleDeg * Math.PI / 180;
    const cos = Math.cos(rad), sin = Math.sin(rad);
    const cx = size / 2, cy = size / 2;
    const rotated = new Float32Array(size * size);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const sx = Math.round(cos * (x - cx) + sin * (y - cy) + cx);
            const sy = Math.round(-sin * (x - cx) + cos * (y - cy) + cy);
            if (sx >= 0 && sx < size && sy >= 0 && sy < size) {
                rotated[y * size + x] = data[sy * size + sx];
            }
        }
    }
    return rotated;
}

function scaleImage(data, scale) {
    const size = 128;
    const cx = size / 2, cy = size / 2;
    const scaled = new Float32Array(size * size);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const sx = Math.round((x - cx) / scale + cx);
            const sy = Math.round((y - cy) / scale + cy);
            if (sx >= 0 && sx < size && sy >= 0 && sy < size) {
                scaled[y * size + x] = data[sy * size + sx];
            }
        }
    }
    return scaled;
}

// ═══ MediaPipe Hands ═══
function initHands() {
    hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`
    });
    hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.7, minTrackingConfidence: 0.5 });
    hands.onResults(onHandResults);
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
        webcamEl.srcObject = stream;
        await webcamEl.play();
        const w = webcamEl.videoWidth || 640;
        const h = webcamEl.videoHeight || 480;
        drawCanvas.width = w; drawCanvas.height = h;
        handCanvas.width = w; handCanvas.height = h;
        cameraRunning = true;
        startOverlay.classList.add('hidden');
        cameraStatus.innerHTML = '<span class="status-dot online"></span><span>Camera On</span>';
        recordingDot.classList.add('active');
        cameraIcon.textContent = '⏹️';
        if (!mpCamera) {
            mpCamera = new Camera(webcamEl, {
                onFrame: async () => { if (cameraRunning && hands) await hands.send({ image: webcamEl }); },
                width: 640, height: 480
            });
        }
        mpCamera.start();
    } catch (err) {
        alert('Camera access denied. Please allow camera and try again.');
    }
}

function stopCamera() {
    cameraRunning = false;
    if (mpCamera) mpCamera.stop();
    const s = webcamEl.srcObject;
    if (s) { s.getTracks().forEach(t => t.stop()); webcamEl.srcObject = null; }
    startOverlay.classList.remove('hidden');
    cameraStatus.innerHTML = '<span class="status-dot offline"></span><span>Camera Off</span>';
    recordingDot.classList.remove('active');
    cameraIcon.textContent = '📷';
    handCtx.clearRect(0, 0, handCanvas.width, handCanvas.height);
}

// ═══ Hand Results (same logic as app.py) ═══
function onHandResults(results) {
    const w = handCanvas.width, h = handCanvas.height;
    handCtx.clearRect(0, 0, w, h);
    handCtx.save();
    handCtx.scale(-1, 1);
    handCtx.translate(-w, 0);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const lm = results.multiHandLandmarks[0];
        drawConnectors(handCtx, lm, HAND_CONNECTIONS, { color: 'rgba(0,212,255,0.4)', lineWidth: 2 });
        drawLandmarks(handCtx, lm, { color: 'rgba(0,255,136,0.6)', lineWidth: 1, radius: 3 });

        const indexUp = lm[8].y < lm[6].y;
        const middleUp = lm[12].y < lm[10].y;
        const ringUp = lm[16].y < lm[14].y;
        const pinkyUp = lm[20].y < lm[18].y;

        if (indexUp && !middleUp && !ringUp && !pinkyUp) {
            const cx = Math.round(lm[8].x * w);
            const cy = Math.round(lm[8].y * h);
            handDisappearedTime = null;

            let shouldAdd = true;
            if (lastPoint) {
                const dx = cx - lastPoint[0], dy = cy - lastPoint[1];
                if (dx*dx + dy*dy < 9) shouldAdd = false;
            }
            if (shouldAdd) { currentPoints.push([cx, cy]); lastPoint = [cx, cy]; }

            handCtx.beginPath();
            handCtx.arc(cx, cy, 12, 0, 2 * Math.PI);
            handCtx.fillStyle = drawColor;
            handCtx.fill();

            drawingStatus.textContent = '✏️ Drawing...';
            drawingStatus.classList.add('active');
        } else {
            handleHandGone();
        }
    } else {
        handleHandGone();
    }

    handCtx.restore();
    redrawCanvas();
    pointCount.textContent = currentPoints.length;
}

function handleHandGone() {
    if (currentPoints.length > MIN_POINTS) {
        if (!handDisappearedTime) {
            handDisappearedTime = Date.now();
            drawingStatus.textContent = '🧠 Classifying...';
            drawingStatus.classList.remove('active');
        }
        if (Date.now() - handDisappearedTime >= CLASSIFY_DELAY) {
            doClassify();
            currentPoints = [];
            lastPoint = null;
            handDisappearedTime = null;
            drawingStatus.textContent = 'Ready to draw';
        }
    } else if (handDisappearedTime && Date.now() - handDisappearedTime > CLASSIFY_DELAY + 500) {
        currentPoints = [];
        lastPoint = null;
        handDisappearedTime = null;
        drawingStatus.textContent = 'Ready to draw';
        drawingStatus.classList.remove('active');
    } else if (!handDisappearedTime && currentPoints.length > 0 && currentPoints.length <= MIN_POINTS) {
        handDisappearedTime = Date.now();
    }
}

async function doClassify() {
    const data = await classifyShape(currentPoints, drawCanvas.width, drawCanvas.height);
    if (!data) return;
    showResult(data);
    if (data.accepted) {
        detectedShapes.push({ name: data.shape, emoji: data.emoji, confidence: data.confidence, color: drawColor, points: [...currentPoints] });
        addHistoryItem(data);
    }
}

// ═══ Canvas Drawing ═══
function redrawCanvas() {
    const w = drawCanvas.width, h = drawCanvas.height;
    drawCtx.clearRect(0, 0, w, h);
    drawCtx.save();
    drawCtx.scale(-1, 1);
    drawCtx.translate(-w, 0);

    for (const shape of detectedShapes) {
        drawCtx.strokeStyle = shape.color;
        drawCtx.lineWidth = 4;
        drawCtx.lineCap = 'round';
        drawCtx.shadowColor = shape.color;
        drawCtx.shadowBlur = 8;
        drawCtx.beginPath();
        for (let i = 0; i < shape.points.length; i++) {
            if (i === 0) drawCtx.moveTo(shape.points[i][0], shape.points[i][1]);
            else drawCtx.lineTo(shape.points[i][0], shape.points[i][1]);
        }
        drawCtx.stroke();
        drawCtx.shadowBlur = 0;
        if (shape.points.length > 0) {
            const lx = shape.points[0][0], ly = shape.points[0][1] - 10;
            drawCtx.font = '600 13px Inter, sans-serif';
            const txt = `${shape.emoji} ${shape.name}`;
            drawCtx.fillStyle = 'rgba(0,0,0,0.7)';
            drawCtx.fillRect(lx - 3, ly - 13, drawCtx.measureText(txt).width + 6, 18);
            drawCtx.fillStyle = '#fff';
            drawCtx.fillText(txt, lx, ly);
        }
    }

    if (currentPoints.length > 1) {
        drawCtx.strokeStyle = drawColor;
        drawCtx.lineWidth = 4;
        drawCtx.lineCap = 'round';
        drawCtx.shadowColor = drawColor;
        drawCtx.shadowBlur = 10;
        drawCtx.beginPath();
        drawCtx.moveTo(currentPoints[0][0], currentPoints[0][1]);
        for (let i = 1; i < currentPoints.length; i++) drawCtx.lineTo(currentPoints[i][0], currentPoints[i][1]);
        drawCtx.stroke();
        drawCtx.shadowBlur = 0;
    }
    drawCtx.restore();
}

// ═══ UI ═══
function showResult(data) {
    const barClass = data.confidence >= 55 ? '' : data.confidence >= 35 ? 'uncertain' : 'low';
    const statusClass = data.accepted ? 'accepted' : 'rejected';
    const statusText = data.accepted ? '✓ Shape Detected' : '? Uncertain';
    resultBody.innerHTML = `
        <div class="result-display">
            <div class="result-emoji">${data.emoji}</div>
            <div class="result-name">${data.shape}</div>
            <div class="result-confidence">${data.confidence}% confidence</div>
            <div class="confidence-bar-container"><div class="confidence-bar ${barClass}" style="width:${data.confidence}%"></div></div>
            <span class="result-status ${statusClass}">${statusText}</span>
        </div>`;
    if (data.top3) {
        predictionsCard.style.display = 'block';
        predictionsBody.innerHTML = data.top3.map((p, i) => `
            <div class="prediction-item">
                <span class="prediction-rank ${i===0?'top':''}">${i+1}</span>
                <div class="prediction-info">
                    <div class="prediction-name">${p.emoji} ${p.name}</div>
                    <div class="prediction-bar"><div class="prediction-bar-fill" style="width:${p.confidence}%"></div></div>
                </div>
                <span class="prediction-conf">${p.confidence}%</span>
            </div>`).join('');
    }
}

function addHistoryItem(data) {
    historyCount.textContent = detectedShapes.length;
    const empty = historyList.querySelector('.empty-state');
    if (empty) empty.remove();
    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = `<span class="history-num">#${detectedShapes.length}</span><span class="history-emoji">${data.emoji}</span><span class="history-name">${data.shape}</span><span class="history-conf">${data.confidence}%</span>`;
    historyList.insertBefore(item, historyList.firstChild);
}

// ═══ Controls ═══
startBtn.addEventListener('click', () => { initHands(); startCamera(); });
toggleCameraBtn.addEventListener('click', () => { if (cameraRunning) stopCamera(); else { if (!hands) initHands(); startCamera(); } });
clearBtn.addEventListener('click', () => {
    currentPoints = []; detectedShapes = []; lastPoint = null; handDisappearedTime = null;
    redrawCanvas(); pointCount.textContent = '0'; historyCount.textContent = '0';
    historyList.innerHTML = '<div class="empty-state small"><p>No shapes detected yet</p></div>';
    resultBody.innerHTML = '<div class="empty-state"><span class="empty-icon">✋</span><p>Draw a shape to see AI predictions</p></div>';
    predictionsCard.style.display = 'none';
});
undoBtn.addEventListener('click', () => {
    if (detectedShapes.length > 0) { detectedShapes.pop(); redrawCanvas(); historyCount.textContent = detectedShapes.length;
        const items = historyList.querySelectorAll('.history-item'); if (items.length > 0) items[0].remove();
        if (detectedShapes.length === 0) historyList.innerHTML = '<div class="empty-state small"><p>No shapes detected yet</p></div>';
    }
});
colorPicker.addEventListener('input', (e) => { drawColor = e.target.value; });

// ═══ Init ═══
loadModel();
