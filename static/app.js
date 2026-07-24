// Backend base URL. Empty string = same-origin (local dev via `python main.py`).
// When frontend and backend live on different domains (Vercel + AWS), set this
// to the backend's HTTPS URL, e.g. "https://api.yourdomain.com".
const API_BASE = "https://16-170-66-162.sslip.io";

const dropZones = document.querySelectorAll('.drop-zone');
const analyzeBtn = document.getElementById('analyze-btn');

const resultsSection = document.getElementById('results-section');
const loadingSpinner = document.getElementById('loading-spinner');
const diagnosisContent = document.getElementById('diagnosis-content');
const errorMessage = document.getElementById('error-message');
const errorText = document.getElementById('error-text');

const accuracyTotal = document.getElementById('accuracy-total');
const accuracyPercent = document.getElementById('accuracy-percent');

// Share Link Elements
const shareLinkSection = document.getElementById('share-link-section');
const shareLinkInput = document.getElementById('share-link-input');
const copyLinkBtn = document.getElementById('copy-link-btn');
const copyStatus = document.getElementById('copy-status');

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

// --- Per-eye upload state ---
const eyeFiles = { left: null, right: null };

function updateAnalyzeButton() {
    analyzeBtn.disabled = !(eyeFiles.left && eyeFiles.right);
}

dropZones.forEach(zone => {
    const eye = zone.dataset.eye;
    const fileInput = zone.querySelector('.file-input');
    const preview = zone.querySelector('.image-preview');
    const content = zone.querySelector('.drop-zone-content');
    const browseBtn = zone.querySelector('.btn-browse');
    const cameraBtnEl = zone.querySelector('.btn-camera');

    browseBtn.addEventListener('click', () => fileInput.click());
    cameraBtnEl.addEventListener('click', () => openCamera(eye));

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        zone.addEventListener(eventName, preventDefaults, false);
    });
    ['dragenter', 'dragover'].forEach(eventName => {
        zone.addEventListener(eventName, () => zone.classList.add('dragover'), false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        zone.addEventListener(eventName, () => zone.classList.remove('dragover'), false);
    });
    zone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) setEyeFile(eye, files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) setEyeFile(eye, e.target.files[0]);
    });
});

function setEyeFile(eye, file) {
    if (!file.type.startsWith('image/')) {
        showError("Please upload a valid image file.");
        return;
    }

    eyeFiles[eye] = file;

    const zone = document.querySelector(`.drop-zone[data-eye="${eye}"]`);
    const preview = zone.querySelector('.image-preview');
    const content = zone.querySelector('.drop-zone-content');

    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.classList.remove('hidden');
        content.classList.add('hidden');
    };
    reader.readAsDataURL(file);

    updateAnalyzeButton();
}

analyzeBtn.addEventListener('click', () => {
    if (eyeFiles.left && eyeFiles.right) {
        processImages(eyeFiles.left, eyeFiles.right);
    }
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// --- Camera Capture Handlers ---
const cameraModal = document.getElementById('camera-modal');
const cameraVideo = document.getElementById('camera-video');
const cameraCanvas = document.getElementById('camera-canvas');
const captureBtn = document.getElementById('capture-btn');
const closeCameraBtn = document.getElementById('close-camera-btn');

let cameraStream = null;
let activeCameraEye = null;

closeCameraBtn.addEventListener('click', closeCamera);
captureBtn.addEventListener('click', capturePhoto);
cameraModal.addEventListener('click', (e) => {
    if (e.target === cameraModal) closeCamera();
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !cameraModal.classList.contains('hidden')) closeCamera();
});

async function openCamera(eye) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError("Camera access is not supported in this browser.");
        return;
    }

    activeCameraEye = eye;

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

    const eye = activeCameraEye;
    cameraCanvas.toBlob((blob) => {
        if (!blob) {
            showError("Failed to capture photo.");
            return;
        }
        const file = new File([blob], `camera-capture-${eye}-${Date.now()}.jpg`, { type: 'image/jpeg' });
        closeCamera();
        setEyeFile(eye, file);
    }, 'image/jpeg', 0.95);
}

// --- Analysis ---
function updateModelCard(eye, modelNum, result) {
    const card = document.querySelector(`.eye-results[data-eye="${eye}"] .model-card[data-model="${modelNum}"]`);
    if (!card) return;

    const diagEl = card.querySelector('[data-field="diag"]');
    diagEl.textContent = result.Diagnosis;
    if (result.Diagnosis === 'Cataract') diagEl.style.color = 'var(--color-cataract)';
    else if (result.Diagnosis === 'Normal') diagEl.style.color = 'var(--color-normal)';
    else diagEl.style.color = 'var(--color-noteye)';

    card.querySelector('[data-field="cataract"]').textContent = result.Cataract;
    card.querySelector('[data-field="normal"]').textContent = result.Normal;
    card.querySelector('[data-field="noteye"]').textContent = result['Not Eye'];
}

async function processImages(leftFile, rightFile) {
    // Reset UI
    resultsSection.classList.remove('hidden');
    diagnosisContent.classList.add('hidden');
    errorMessage.classList.add('hidden');
    loadingSpinner.classList.remove('hidden');
    shareLinkSection.classList.add('hidden');
    copyStatus.classList.add('hidden');

    const formData = new FormData();
    formData.append('left_file', leftFile);
    formData.append('right_file', rightFile);

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });

        let data;
        try {
            data = await response.json();
        } catch (parseErr) {
            // Response wasn't JSON at all — e.g. a proxy/server error page
            // (too-large upload, gateway timeout, etc.) rather than our API.
            showError(
                response.status === 413
                    ? "One or both images are too large. Please use smaller photos."
                    : `Server error (${response.status}). Please try again.`
            );
            return;
        }

        if (!response.ok || data.error) {
            showError(data.error || "Failed to analyze the images.");
            return;
        }

        loadingSpinner.classList.add('hidden');
        diagnosisContent.classList.remove('hidden');

        ['left', 'right'].forEach(eye => {
            data[eye].models.forEach(m => updateModelCard(eye, m.model, m.result));
        });

        if (data.share_token) {
            const base = API_BASE || window.location.origin;
            shareLinkInput.value = `${base}/review/${data.share_token}`;
            shareLinkSection.classList.remove('hidden');
        }

        refreshAccuracy();
    } catch (err) {
        showError("Failed to connect to the AI engine.");
        console.error(err);
    }
}

function showError(msg) {
    loadingSpinner.classList.add('hidden');
    errorMessage.classList.remove('hidden');
    errorText.textContent = msg;
}
