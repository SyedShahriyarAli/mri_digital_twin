// Security Testing Dashboard JavaScript

let currentScan = null;
let originalData = null;
let tamperedData = null;
let isAttackApplied = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Security Testing Dashboard loaded');
    loadAvailableScans();
    initializeSlider();
});

// Load available scans
async function loadAvailableScans() {
    try {
        const response = await fetch('/api/scans');
        const data = await response.json();
        
        const selectElement = document.getElementById('scan-select');
        selectElement.innerHTML = '<option value="">-- Select a scan to test --</option>';
        
        data.scans.forEach(scan => {
            const option = document.createElement('option');
            option.value = JSON.stringify({ name: scan.name, format: scan.format });
            option.textContent = `${scan.name} (${scan.format})`;
            selectElement.appendChild(option);
        });
        
    } catch (error) {
        console.error('Error loading scans:', error);
        showNotification('Failed to load scans', 'error');
    }
}

// Load scan for testing
async function loadScanForTesting() {
    const selectElement = document.getElementById('scan-select');
    const selectedValue = selectElement.value;
    
    if (!selectedValue) {
        showNotification('Please select a scan', 'warning');
        return;
    }
    
    try {
        currentScan = JSON.parse(selectedValue);
        
        // Show loading
        document.getElementById('original-spinner').style.display = 'flex';
        document.getElementById('tampered-spinner').style.display = 'flex';
        
        // Load the scan
        const response = await fetch('/api/security/load_scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                scan_name: currentScan.name,
                format: currentScan.format
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Store original data
        originalData = data;
        tamperedData = data;
        isAttackApplied = false;
        
        // Update UI
        document.getElementById('empty-state').style.display = 'none';
        document.getElementById('testing-area').style.display = 'block';
        
        // Load images
        loadSliceImages(data.middle_slice);
        
        // Update metadata
        updateMetadata(data.metadata, data.metadata);
        
        // Setup slider
        const slider = document.getElementById('slice-slider');
        slider.max = data.dimensions[0] - 1;
        slider.value = data.middle_slice;
        slider.disabled = false;
        document.getElementById('slice-display').textContent = data.middle_slice;
        
        // Enable attack button
        document.getElementById('inject-attack-btn').disabled = false;
        
        // Reset buttons
        document.getElementById('run-detection-btn').disabled = true;
        document.getElementById('reset-btn').disabled = true;
        
        // Clear detection results
        document.getElementById('detection-section').style.display = 'none';
        
        // Update status
        updateIntegrityStatus('secure');
        
        // Add log entry
        addLogEntry('Scan loaded successfully', 'detection');
        
        showNotification('Scan loaded successfully', 'success');
        
    } catch (error) {
        console.error('Error loading scan:', error);
        showNotification('Failed to load scan: ' + error.message, 'error');
        document.getElementById('original-spinner').style.display = 'none';
        document.getElementById('tampered-spinner').style.display = 'none';
    }
}

// Load slice images
function loadSliceImages(sliceIndex) {
    const originalImg = document.getElementById('original-image');
    const tamperedImg = document.getElementById('tampered-image');
    
    // Show spinners
    document.getElementById('original-spinner').style.display = 'flex';
    document.getElementById('tampered-spinner').style.display = 'flex';
    
    const timestamp = new Date().getTime();
    const originalPath = `/api/security/get_original_slice/${currentScan.name}/axial/${sliceIndex}?t=${timestamp}`;
    const tamperedPath = `/api/security/get_tampered_slice/${currentScan.name}/axial/${sliceIndex}?t=${timestamp}`;
    
    originalImg.src = originalPath;
    tamperedImg.src = tamperedPath;
    
    originalImg.onload = () => {
        document.getElementById('original-spinner').style.display = 'none';
    };
    
    originalImg.onerror = () => {
        document.getElementById('original-spinner').style.display = 'none';
        console.error('Failed to load original image');
    };
    
    tamperedImg.onload = () => {
        document.getElementById('tampered-spinner').style.display = 'none';
    };
    
    tamperedImg.onerror = () => {
        document.getElementById('tampered-spinner').style.display = 'none';
        console.error('Failed to load tampered image');
    };
    
    document.getElementById('original-slice').textContent = sliceIndex;
    document.getElementById('tampered-slice').textContent = sliceIndex;
}

// Initialize slider
function initializeSlider() {
    const slider = document.getElementById('slice-slider');
    slider.addEventListener('input', function() {
        const sliceIndex = parseInt(this.value);
        document.getElementById('slice-display').textContent = sliceIndex;
        loadSliceImages(sliceIndex);
    });
}

// Inject attack
async function injectAttack() {
    const attackType = document.getElementById('attack-type').value;
    const intensity = document.getElementById('attack-intensity').value;
    
    if (!attackType) {
        showNotification('Please select an attack type', 'warning');
        return;
    }
    
    if (!currentScan) {
        showNotification('Please load a scan first', 'warning');
        return;
    }
    
    try {
        // Show loading
        document.getElementById('tampered-spinner').style.display = 'flex';
        
        const response = await fetch('/api/security/inject_attack', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                scan_name: currentScan.name,
                attack_type: attackType,
                intensity: intensity
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Update state
        isAttackApplied = true;
        tamperedData = data;
        
        // Reload current slice
        const currentSlice = parseInt(document.getElementById('slice-slider').value);
        loadSliceImages(currentSlice);
        
        // Update metadata
        updateMetadata(originalData.metadata, data.metadata);
        
        // Update badge
        document.getElementById('tamper-badge').textContent = 'Modified';
        document.getElementById('tamper-badge').className = 'badge badge-danger';
        
        // Enable detection button
        document.getElementById('run-detection-btn').disabled = false;
        document.getElementById('reset-btn').disabled = false;
        
        // Update status
        updateIntegrityStatus('warning');
        
        // Add log entry
        addLogEntry(`Attack injected: ${attackType} (${intensity})`, 'attack');
        
        showNotification('Attack injected successfully', 'success');
        
    } catch (error) {
        console.error('Error injecting attack:', error);
        showNotification('Failed to inject attack: ' + error.message, 'error');
        document.getElementById('tampered-spinner').style.display = 'none';
    }
}

// Run detection
async function runDetection() {
    if (!currentScan || !isAttackApplied) {
        showNotification('No attack to detect', 'warning');
        return;
    }
    
    try {
        // Show detection section
        document.getElementById('detection-section').style.display = 'block';
        
        // Reset detection statuses
        resetDetectionStatuses();
        
        // Add log entry
        addLogEntry('Running security detection...', 'detection');
        
        // Run detection
        const response = await fetch('/api/security/run_detection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                scan_name: currentScan.name
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Update detection results
        updateDetectionResults(data.results);
        
        // Update overall assessment
        updateAssessment(data.assessment);
        
        // Update integrity status
        updateIntegrityStatus(data.assessment.status);
        
        // Add log entry
        addLogEntry(`Detection complete: ${data.assessment.message}`, 'detection');
        
        showNotification('Detection completed', 'success');
        
    } catch (error) {
        console.error('Error running detection:', error);
        showNotification('Failed to run detection: ' + error.message, 'error');
    }
}

// Reset scan
async function resetScan() {
    if (!currentScan) {
        return;
    }
    
    try {
        const response = await fetch('/api/security/reset_scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                scan_name: currentScan.name
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Reset state
        isAttackApplied = false;
        tamperedData = originalData;
        
        // Reload current slice
        const currentSlice = parseInt(document.getElementById('slice-slider').value);
        loadSliceImages(currentSlice);
        
        // Update metadata
        updateMetadata(originalData.metadata, originalData.metadata);
        
        // Update badge
        document.getElementById('tamper-badge').textContent = 'Normal';
        document.getElementById('tamper-badge').className = 'badge badge-info';
        
        // Disable buttons
        document.getElementById('run-detection-btn').disabled = true;
        document.getElementById('reset-btn').disabled = true;
        
        // Hide detection results
        document.getElementById('detection-section').style.display = 'none';
        
        // Update status
        updateIntegrityStatus('secure');
        
        // Add log entry
        addLogEntry('Scan reset to original state', 'reset');
        
        showNotification('Scan reset successfully', 'success');
        
    } catch (error) {
        console.error('Error resetting scan:', error);
        showNotification('Failed to reset scan: ' + error.message, 'error');
    }
}

// Update metadata comparison
function updateMetadata(original, current) {
    // Original metadata
    document.getElementById('orig-date').textContent = original.scan_date || '-';
    document.getElementById('orig-scanner').textContent = original.scanner || '-';
    document.getElementById('orig-tr').textContent = original.tr || '-';
    document.getElementById('orig-te').textContent = original.te || '-';
    document.getElementById('orig-hash').textContent = original.hash || '-';
    
    // Current metadata
    document.getElementById('curr-date').textContent = current.scan_date || '-';
    document.getElementById('curr-scanner').textContent = current.scanner || '-';
    document.getElementById('curr-tr').textContent = current.tr || '-';
    document.getElementById('curr-te').textContent = current.te || '-';
    document.getElementById('curr-hash').textContent = current.hash || '-';
    
    // Highlight changes
    highlightChanges('date', original.scan_date, current.scan_date);
    highlightChanges('scanner', original.scanner, current.scanner);
    highlightChanges('tr', original.tr, current.tr);
    highlightChanges('te', original.te, current.te);
    highlightChanges('hash', original.hash, current.hash);
}

function highlightChanges(field, original, current) {
    const element = document.getElementById(`curr-${field}`);
    if (original !== current) {
        element.classList.add('changed');
        element.classList.remove('valid');
    } else {
        element.classList.remove('changed');
        element.classList.add('valid');
    }
}

// Update integrity status
function updateIntegrityStatus(status) {
    const statusCard = document.getElementById('integrity-status');
    const statusValue = statusCard.querySelector('.status-value');
    
    statusCard.className = 'status-card';
    
    if (status === 'secure') {
        statusCard.classList.add('secure');
        statusValue.textContent = 'SECURE';
    } else if (status === 'compromised') {
        statusCard.classList.add('compromised');
        statusValue.textContent = 'COMPROMISED';
    } else if (status === 'warning') {
        statusCard.classList.add('warning');
        statusValue.textContent = 'WARNING';
    }
}

// Reset detection statuses
function resetDetectionStatuses() {
    const statuses = ['hash-status', 'anomaly-status', 'metadata-status'];
    statuses.forEach(id => {
        const element = document.getElementById(id);
        element.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Checking...</span>';
        element.className = 'detection-status';
    });
}

// Update detection results
function updateDetectionResults(results) {
    // Hash verification
    updateDetectionStatus('hash', results.hash_verification);
    
    // Anomaly detection
    updateDetectionStatus('anomaly', results.anomaly_detection);
    
    // Metadata validation
    updateDetectionStatus('metadata', results.metadata_validation);
}

function updateDetectionStatus(type, result) {
    const statusElement = document.getElementById(`${type}-status`);
    const detailsElement = document.getElementById(`${type}-details`);
    
    const icon = result.passed 
        ? '<i class="fas fa-check-circle"></i>' 
        : '<i class="fas fa-times-circle"></i>';
    
    statusElement.innerHTML = `${icon}<span>${result.status}</span>`;
    statusElement.className = `detection-status ${result.passed ? 'success' : 'failed'}`;
    
    detailsElement.textContent = result.message;
}

// Update assessment
function updateAssessment(assessment) {
    const card = document.getElementById('assessment-card');
    const title = document.getElementById('assessment-title');
    const message = document.getElementById('assessment-message');
    const icon = document.getElementById('assessment-icon');
    
    card.className = 'assessment-card';
    
    if (assessment.status === 'secure') {
        card.classList.add('secure');
        icon.innerHTML = '<i class="fas fa-shield-alt"></i>';
        title.textContent = 'System Secure';
    } else if (assessment.status === 'compromised') {
        card.classList.add('compromised');
        icon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
        title.textContent = 'Security Breach Detected';
    } else if (assessment.status === 'warning') {
        card.classList.add('warning');
        icon.innerHTML = '<i class="fas fa-exclamation-circle"></i>';
        title.textContent = 'Warning';
    }
    
    message.textContent = assessment.message;
}

// Add log entry
function addLogEntry(message, type) {
    const container = document.getElementById('log-container');
    
    // Remove empty state if exists
    const emptyState = container.querySelector('.log-empty');
    if (emptyState) {
        emptyState.remove();
    }
    
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    
    const now = new Date();
    const timeStr = now.toLocaleTimeString();
    
    let iconClass = 'fa-info-circle';
    let iconType = '';
    
    if (type === 'attack') {
        iconClass = 'fa-biohazard';
        iconType = 'attack';
    } else if (type === 'detection') {
        iconClass = 'fa-search';
        iconType = 'detection';
    } else if (type === 'reset') {
        iconClass = 'fa-undo';
        iconType = 'reset';
    }
    
    entry.innerHTML = `
        <span class="log-time">${timeStr}</span>
        <i class="fas ${iconClass} log-icon ${iconType}"></i>
        <span class="log-message">${message}</span>
    `;
    
    container.insertBefore(entry, container.firstChild);
}

// Show notification
function showNotification(message, type) {
    // Simple console notification for now
    // You can implement a toast notification library here
    console.log(`[${type.toUpperCase()}] ${message}`);
    
    // Could use a library like toastr or create custom notifications
}
