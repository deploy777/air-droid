/**
 * AI Air Canvas — Frontend
 * All processing happens server-side (same as original app.py).
 * JS just controls start/stop and polls for results.
 */

const videoFeed = document.getElementById('videoFeed');
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

let cameraRunning = false;
let pollInterval = null;
let lastResultId = 0;
let historyItems = [];

// ═══ Splash Screen ═══
window.addEventListener('load', () => {
    setTimeout(() => {
        const splash = document.getElementById('splash');
        splash.classList.add('fade-out');
        setTimeout(() => {
            splash.style.display = 'none';
            document.getElementById('app').classList.remove('hidden');
        }, 600);
    }, 2800);
});

// ═══ Camera Start/Stop ═══
async function startCamera() {
    try {
        await fetch('/api/start', { method: 'POST' });
        cameraRunning = true;

        // Start video stream
        videoFeed.src = '/video_feed';
        videoFeed.style.display = 'block';

        startOverlay.classList.add('hidden');
        cameraStatus.innerHTML = '<span class="status-dot online"></span><span>Camera On</span>';
        recordingDot.classList.add('active');
        cameraIcon.textContent = '⏹️';

        // Start polling for state updates
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(pollState, 500);
    } catch (err) {
        console.error('Start error:', err);
        alert('Failed to start camera. Make sure a webcam is connected.');
    }
}

async function stopCamera() {
    try {
        await fetch('/api/stop', { method: 'POST' });
    } catch (e) {}
    cameraRunning = false;
    videoFeed.src = '';
    videoFeed.style.display = 'none';
    startOverlay.classList.remove('hidden');
    cameraStatus.innerHTML = '<span class="status-dot offline"></span><span>Camera Off</span>';
    recordingDot.classList.remove('active');
    cameraIcon.textContent = '📷';
    drawingStatus.textContent = 'Ready to draw';
    drawingStatus.classList.remove('active');

    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// ═══ Poll Server State ═══
async function pollState() {
    try {
        const res = await fetch('/api/state');
        const data = await res.json();

        // Update point count
        pointCount.textContent = data.num_points;

        // Drawing indicator
        if (data.num_points > 5) {
            drawingStatus.textContent = '✏️ Drawing...';
            drawingStatus.classList.add('active');
        } else {
            drawingStatus.textContent = 'Ready to draw';
            drawingStatus.classList.remove('active');
        }

        // Update result if new
        if (data.last_result && JSON.stringify(data.last_result) !== lastResultId) {
            lastResultId = JSON.stringify(data.last_result);
            showResult(data.last_result);
        }

        // Update history
        if (data.shapes.length !== historyItems.length) {
            historyItems = data.shapes;
            updateHistory();
        }

        // Show top predictions
        if (data.top_predictions && data.top_predictions.length > 0) {
            showPredictions(data.top_predictions);
        }
    } catch (err) {
        // Silently retry
    }
}

// ═══ UI Updates ═══
function showResult(data) {
    if (!data || data.shape === 'Unknown') {
        resultBody.innerHTML = `
            <div class="result-display">
                <div class="result-emoji">❓</div>
                <div class="result-name">Not recognized</div>
                <div class="result-confidence">${data ? data.confidence : 0}% confidence</div>
            </div>`;
        return;
    }

    const barClass = data.confidence >= 55 ? '' : data.confidence >= 35 ? 'uncertain' : 'low';
    const statusClass = data.accepted ? 'accepted' : 'rejected';
    const statusText = data.accepted ? '✓ Shape Detected' : '? Uncertain';

    resultBody.innerHTML = `
        <div class="result-display">
            <div class="result-emoji">${data.emoji}</div>
            <div class="result-name">${data.shape}</div>
            <div class="result-confidence">${data.confidence}% confidence</div>
            <div class="confidence-bar-container">
                <div class="confidence-bar ${barClass}" style="width: ${data.confidence}%"></div>
            </div>
            <span class="result-status ${statusClass}">${statusText}</span>
        </div>`;

    resultCard.style.borderColor = data.accepted ? 'rgba(0,255,136,0.3)' : 'rgba(255,170,0,0.3)';
    resultCard.style.boxShadow = data.accepted
        ? '0 0 30px rgba(0,255,136,0.15)'
        : '0 0 30px rgba(255,170,0,0.1)';

    if (data.top3 && data.top3.length > 0) {
        showPredictions(data.top3);
    }
}

function showPredictions(preds) {
    predictionsCard.style.display = 'block';
    predictionsBody.innerHTML = preds.map((p, i) => `
        <div class="prediction-item">
            <span class="prediction-rank ${i === 0 ? 'top' : ''}">${i + 1}</span>
            <div class="prediction-info">
                <div class="prediction-name">${p.emoji} ${p.name}</div>
                <div class="prediction-bar">
                    <div class="prediction-bar-fill" style="width: ${p.confidence}%"></div>
                </div>
            </div>
            <span class="prediction-conf">${p.confidence}%</span>
        </div>`).join('');
}

function updateHistory() {
    historyCount.textContent = historyItems.length;
    if (historyItems.length === 0) {
        historyList.innerHTML = '<div class="empty-state small"><p>No shapes detected yet</p></div>';
        return;
    }
    historyList.innerHTML = historyItems.map((s, i) => `
        <div class="history-item">
            <span class="history-num">#${i + 1}</span>
            <span class="history-emoji">${s.emoji}</span>
            <span class="history-name">${s.name}</span>
            <span class="history-conf">${s.confidence}%</span>
        </div>`).reverse().join('');
}

// ═══ Controls ═══
startBtn.addEventListener('click', startCamera);

toggleCameraBtn.addEventListener('click', () => {
    if (cameraRunning) stopCamera();
    else startCamera();
});

clearBtn.addEventListener('click', async () => {
    await fetch('/api/clear', { method: 'POST' });
    historyItems = [];
    updateHistory();
    pointCount.textContent = '0';
    resultBody.innerHTML = '<div class="empty-state"><span class="empty-icon">✋</span><p>Draw a shape to see AI predictions</p></div>';
    predictionsCard.style.display = 'none';
    resultCard.style.borderColor = 'var(--border-glow)';
    resultCard.style.boxShadow = 'var(--shadow-glow)';
    lastResultId = 0;
});

undoBtn.addEventListener('click', async () => {
    await fetch('/api/undo', { method: 'POST' });
});

colorPicker.addEventListener('input', async (e) => {
    await fetch('/api/color', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ color: e.target.value })
    });
});
