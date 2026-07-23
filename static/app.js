// Backend base URL. Empty string = same-origin (local dev via `python main.py`).
// When frontend and backend live on different domains (Vercel + AWS), set this
// to the backend's HTTPS URL, e.g. "https://api.yourdomain.com".
const API_BASE = "https://16-170-66-162.sslip.io";

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const dropZoneContent = document.querySelector('.drop-zone-content');

const resultsSection = document.getElementById('results-section');
const loadingSpinner = document.getElementById('loading-spinner');
const diagnosisContent = document.getElementById('diagnosis-content');
const errorMessage = document.getElementById('error-message');
const errorText = document.getElementById('error-text');

// Doctor Feedback Elements
const feedbackSection = document.getElementById('feedback-section');
const feedbackCataractBtn = document.getElementById('feedback-cataract');
const feedbackNormalBtn = document.getElementById('feedback-normal');
const feedbackStatus = document.getElementById('feedback-status');
const accuracyTotal = document.getElementById('accuracy-total');
const accuracyPercent = document.getElementById('accuracy-percent');

// Share Link Elements
const shareLinkSection = document.getElementById('share-link-section');
const shareLinkInput = document.getElementById('share-link-input');
const copyLinkBtn = document.getElementById('copy-link-btn');
const copyStatus = document.getElementById('copy-status');

let currentPredictionId = null;

feedbackCataractBtn.addEventListener('click', () => submitFeedback('Cataract'));
feedbackNormalBtn.addEventListener('click', () => submitFeedback('Normal'));

copyLinkBtn.addEventListener('click', async () => {
    try {
        await navigator.clipboard.writeText(shareLinkInput.value);
        copyStatus.textContent = "Link copied to clipboard.";
    } catch (err) {
        shareLinkInput.select();
        copyStatus.textContent = "Press Ctrl+C to copy (clipboard access blocked).";
    }
    copyStatus.classList.remove('hidden');
});

async function submitFeedback(label) {
    if (!currentPredictionId) return;

    feedbackCataractBtn.disabled = true;
    feedbackNormalBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: currentPredictionId, doctor_label: label })
        });

        if (!response.ok) throw new Error('Failed to save feedback');

        (label === 'Cataract' ? feedbackCataractBtn : feedbackNormalBtn).classList.add('selected');
        feedbackStatus.textContent = `Saved as "${label}" — thank you, this will help fine-tune the model.`;
        feedbackStatus.classList.remove('hidden');

        refreshAccuracy();
    } catch (err) {
        console.error(err);
        feedbackStatus.textContent = "Failed to save your feedback. Please try again.";
        feedbackStatus.classList.remove('hidden');
        feedbackCataractBtn.disabled = false;
        feedbackNormalBtn.disabled = false;
    }
}

async function refreshAccuracy() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const data = await response.json();
        const model3Accuracy = data.per_model_accuracy ? data.per_model_accuracy.model3 : null;
        accuracyTotal.textContent = data.total_labeled;
        accuracyPercent.textContent = model3Accuracy !== null && model3Accuracy !== undefined ? `${model3Accuracy}%` : '—';
    } catch (err) {
        console.error('Failed to load accuracy stats:', err);
    }
}

refreshAccuracy();

// Diagnosis Elements
// (Removed old ensemble elements)

// --- Camera Capture Handlers ---
const cameraBtn = document.getElementById('camera-btn');
const cameraModal = document.getElementById('camera-modal');
const cameraVideo = document.getElementById('camera-video');
const cameraCanvas = document.getElementById('camera-canvas');
const captureBtn = document.getElementById('capture-btn');
const closeCameraBtn = document.getElementById('close-camera-btn');

let cameraStream = null;

cameraBtn.addEventListener('click', openCamera);
closeCameraBtn.addEventListener('click', closeCamera);
captureBtn.addEventListener('click', capturePhoto);
cameraModal.addEventListener('click', (e) => {
    if (e.target === cameraModal) closeCamera();
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !cameraModal.classList.contains('hidden')) closeCamera();
});

async function openCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError("Camera access is not supported in this browser.");
        return;
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' },
            audio: false
        });

        // Flashlight/torch must never be turned on, even if the device supports it.
        const track = cameraStream.getVideoTracks()[0];
        const capabilities = track.getCapabilities ? track.getCapabilities() : {};
        if (capabilities.torch) {
            try {
                await track.applyConstraints({ advanced: [{ torch: false }] });
            } catch (e) {
                console.warn('Could not explicitly disable torch:', e);
            }
        }

        cameraVideo.srcObject = cameraStream;
        cameraModal.classList.remove('hidden');
    } catch (err) {
        console.error(err);
        showError("Unable to access camera. Please check permissions.");
    }
}

function closeCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    cameraVideo.srcObject = null;
    cameraModal.classList.add('hidden');
}

function capturePhoto() {
    const width = cameraVideo.videoWidth;
    const height = cameraVideo.videoHeight;
    if (!width || !height) {
        showError("Camera is not ready yet. Please wait a moment and try again.");
        return;
    }

    cameraCanvas.width = width;
    cameraCanvas.height = height;
    cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0, width, height);

    cameraCanvas.toBlob((blob) => {
        if (!blob) {
            showError("Failed to capture photo.");
            return;
        }
        const file = new File([blob], `camera-capture-${Date.now()}.jpg`, { type: 'image/jpeg' });
        closeCamera();
        handleFile(file);
    }, 'image/jpeg', 0.95);
}

// --- Drag and Drop Handlers ---
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', handleDrop, false);
fileInput.addEventListener('change', handleFileSelect, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) handleFile(files[0]);
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError("Please upload a valid image file.");
        return;
    }

    // Show Image Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.classList.remove('hidden');
        dropZoneContent.classList.add('hidden');
    }
    reader.readAsDataURL(file);

    // Process with AI
    processImage(file);
}

async function processImage(file) {
    // Reset UI
    resultsSection.classList.remove('hidden');
    diagnosisContent.classList.add('hidden');
    errorMessage.classList.add('hidden');
    loadingSpinner.classList.remove('hidden');

    currentPredictionId = null;
    feedbackSection.classList.add('hidden');
    feedbackStatus.classList.add('hidden');
    feedbackCataractBtn.disabled = false;
    feedbackNormalBtn.disabled = false;
    feedbackCataractBtn.classList.remove('selected');
    feedbackNormalBtn.classList.remove('selected');
    shareLinkSection.classList.add('hidden');
    copyStatus.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        
        // Show the results section but clear previous data
        loadingSpinner.classList.add('hidden');
        diagnosisContent.classList.remove('hidden');
        [1, 2, 3].forEach(m => {
            document.getElementById(`val-exp${m}-diag`).textContent = "Thinking...";
            document.getElementById(`val-exp${m}-diag`).style.color = "var(--text-main)";
            document.getElementById(`val-exp${m}-c`).textContent = "0";
            document.getElementById(`val-exp${m}-n`).textContent = "0";
            document.getElementById(`val-exp${m}-noteye`).textContent = "0";
        });

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');
            
            for (let line of lines) {
                if (!line.trim()) continue;
                const data = JSON.parse(line);
                
                if (data.error) {
                    showError(data.error);
                    return;
                }

                if (data.done) {
                    currentPredictionId = data.id;
                    feedbackSection.classList.remove('hidden');

                    if (data.share_token) {
                        const base = API_BASE || window.location.origin;
                        shareLinkInput.value = `${base}/review/${data.share_token}`;
                        shareLinkSection.classList.remove('hidden');
                    }
                    continue;
                }

                if (data.model) {
                    const exp = `exp${data.model}`;
                    const diagVal = document.getElementById(`val-${exp}-diag`);
                    diagVal.textContent = data.result.Diagnosis;
                    
                    if (data.result.Diagnosis === 'Cataract') diagVal.style.color = 'var(--color-cataract)';
                    else if (data.result.Diagnosis === 'Normal') diagVal.style.color = 'var(--color-normal)';
                    else diagVal.style.color = 'var(--color-noteye)';

                    document.getElementById(`val-${exp}-c`).textContent = data.result.Cataract;
                    document.getElementById(`val-${exp}-n`).textContent = data.result.Normal;
                    document.getElementById(`val-${exp}-noteye`).textContent = data.result['Not Eye'] || data.result.NotEye || 0;
                }
            }
        }

    } catch (err) {
        showError("Failed to connect to the AI engine.");
        console.error(err);
    }
}

// (displayResults function is removed since it's handled in the stream loop)

function showError(msg) {
    loadingSpinner.classList.add('hidden');
    errorMessage.classList.remove('hidden');
    errorText.textContent = msg;
}
