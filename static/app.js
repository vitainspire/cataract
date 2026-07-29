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
    openCropModal(eye, file);
}

function finalizeEyeFile(eye, file) {
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

// --- Crop Modal ---
const cropModal = document.getElementById('crop-modal');
const cropTitle = document.getElementById('crop-title');
const cropStage = document.getElementById('crop-stage');
const cropImage = document.getElementById('crop-image');
const cropBox = document.getElementById('crop-box');
const cropCanvas = document.getElementById('crop-canvas');
const cropConfirmBtn = document.getElementById('crop-confirm-btn');
const cropSkipBtn = document.getElementById('crop-skip-btn');

const CROP_MIN_SIZE = 40;

let cropEye = null;
let cropOriginalFile = null;
let cropBoxRect = { left: 0, top: 0, size: 0 };
let cropDrag = null; // { mode: 'move'|'resize', handle, startX, startY, startRect }

function openCropModal(eye, file) {
    cropEye = eye;
    cropOriginalFile = file;
    cropTitle.textContent = `Adjust ${eye === 'left' ? 'Left' : 'Right'} Eye Crop`;

    const reader = new FileReader();
    reader.onload = (e) => {
        cropImage.src = e.target.result;
    };
    reader.readAsDataURL(file);

    cropModal.classList.remove('hidden');
}

function closeCropModal() {
    cropModal.classList.add('hidden');
    cropImage.src = '';
    cropEye = null;
    cropOriginalFile = null;
}

cropImage.addEventListener('load', () => {
    const stageWidth = cropStage.clientWidth;
    const stageHeight = cropStage.clientHeight;
    const size = Math.max(CROP_MIN_SIZE, Math.min(stageWidth, stageHeight) * 0.7);
    cropBoxRect = {
        left: (stageWidth - size) / 2,
        top: (stageHeight - size) / 2,
        size
    };
    renderCropBox();
});

function renderCropBox() {
    cropBox.style.left = `${cropBoxRect.left}px`;
    cropBox.style.top = `${cropBoxRect.top}px`;
    cropBox.style.width = `${cropBoxRect.size}px`;
    cropBox.style.height = `${cropBoxRect.size}px`;
}

cropBox.addEventListener('pointerdown', (e) => {
    const handle = e.target.dataset.handle;
    cropDrag = {
        mode: handle ? 'resize' : 'move',
        handle,
        startX: e.clientX,
        startY: e.clientY,
        startRect: { ...cropBoxRect }
    };
    e.preventDefault();
});

document.addEventListener('pointermove', (e) => {
    if (!cropDrag) return;

    const stageWidth = cropStage.clientWidth;
    const stageHeight = cropStage.clientHeight;
    const dx = e.clientX - cropDrag.startX;
    const dy = e.clientY - cropDrag.startY;
    const start = cropDrag.startRect;

    if (cropDrag.mode === 'move') {
        cropBoxRect.left = Math.max(0, Math.min(stageWidth - start.size, start.left + dx));
        cropBoxRect.top = Math.max(0, Math.min(stageHeight - start.size, start.top + dy));
    } else {
        const px = e.clientX - cropStage.getBoundingClientRect().left;
        const py = e.clientY - cropStage.getBoundingClientRect().top;
        let anchorX, anchorY, newLeft, newTop, size;

        if (cropDrag.handle === 'br') {
            anchorX = start.left;
            anchorY = start.top;
            size = Math.max(CROP_MIN_SIZE, Math.max(px - anchorX, py - anchorY));
            size = Math.min(size, stageWidth - anchorX, stageHeight - anchorY);
            newLeft = anchorX;
            newTop = anchorY;
        } else if (cropDrag.handle === 'tl') {
            anchorX = start.left + start.size;
            anchorY = start.top + start.size;
            size = Math.max(CROP_MIN_SIZE, Math.max(anchorX - px, anchorY - py));
            size = Math.min(size, anchorX, anchorY);
            newLeft = anchorX - size;
            newTop = anchorY - size;
        } else if (cropDrag.handle === 'tr') {
            anchorX = start.left;
            anchorY = start.top + start.size;
            size = Math.max(CROP_MIN_SIZE, Math.max(px - anchorX, anchorY - py));
            size = Math.min(size, stageWidth - anchorX, anchorY);
            newLeft = anchorX;
            newTop = anchorY - size;
        } else if (cropDrag.handle === 'bl') {
            anchorX = start.left + start.size;
            anchorY = start.top;
            size = Math.max(CROP_MIN_SIZE, Math.max(anchorX - px, py - anchorY));
            size = Math.min(size, anchorX, stageHeight - anchorY);
            newLeft = anchorX - size;
            newTop = anchorY;
        } else {
            return;
        }

        cropBoxRect = { left: newLeft, top: newTop, size };
    }

    renderCropBox();
});

document.addEventListener('pointerup', () => {
    cropDrag = null;
});

cropConfirmBtn.addEventListener('click', () => {
    const stageWidth = cropStage.clientWidth;
    const scale = cropImage.naturalWidth / stageWidth;

    const sx = cropBoxRect.left * scale;
    const sy = cropBoxRect.top * scale;
    const sSize = cropBoxRect.size * scale;

    cropCanvas.width = Math.round(sSize);
    cropCanvas.height = Math.round(sSize);
    const ctx = cropCanvas.getContext('2d');
    ctx.drawImage(cropImage, sx, sy, sSize, sSize, 0, 0, sSize, sSize);

    const eye = cropEye;
    cropCanvas.toBlob((blob) => {
        if (!blob) {
            showError("Failed to crop the image.");
            return;
        }
        const file = new File([blob], `${eye}-cropped-${Date.now()}.jpg`, { type: 'image/jpeg' });
        closeCropModal();
        finalizeEyeFile(eye, file);
    }, 'image/jpeg', 0.95);
});

cropSkipBtn.addEventListener('click', () => {
    const eye = cropEye;
    const file = cropOriginalFile;
    closeCropModal();
    finalizeEyeFile(eye, file);
});

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
            // Use this site's own domain, not the backend's — Vercel transparently
            // proxies /review/* to the backend (see vercel.json), so the link a
            // doctor gets shows the Vercel domain instead of the raw backend URL.
            shareLinkInput.value = `${window.location.origin}/review/${data.share_token}`;
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
