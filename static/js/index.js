// MRI Digital Twin - Dashboard JavaScript

// State
let allScans = [];
let filteredScans = [];
let currentScan = null;
let currentFormat = null;
let dimensions = {};
let middleIndices = {};

// DOM Elements
const scansGrid = document.getElementById('scans-grid');
const loadingState = document.getElementById('loading-state');
const emptyState = document.getElementById('empty-state');
const searchInput = document.getElementById('search-input');
const formatFilter = document.getElementById('format-filter');
const refreshBtn = document.getElementById('refresh-btn');
const totalScansEl = document.getElementById('total-scans');
const viewerModal = document.getElementById('viewer-modal');

// Viewer elements
const views = ['axial', 'sagittal', 'coronal'];
const sliders = {};
const images = {};
const spinners = {};
const currentSpans = {};
const totalSpans = {};

// Initialize viewer references
views.forEach(view =>
{
    sliders[view] = document.getElementById(`${view}-slider`);
    images[view] = document.getElementById(`${view}-image`);
    spinners[view] = document.getElementById(`${view}-spinner`);
    currentSpans[view] = document.getElementById(`${view}-current`);
    totalSpans[view] = document.getElementById(`${view}-total`);
});

// Load scans on page load
document.addEventListener('DOMContentLoaded', () =>
{
    loadScans();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners()
{
    searchInput.addEventListener('input', filterScans);
    formatFilter.addEventListener('change', filterScans);
    refreshBtn.addEventListener('click', loadScans);

    // Slider event handlers
    views.forEach(view =>
    {
        sliders[view].addEventListener('input', (e) =>
        {
            const sliceIndex = parseInt(e.target.value);
            currentSpans[view].textContent = sliceIndex;
        });

        sliders[view].addEventListener('change', (e) =>
        {
            const sliceIndex = parseInt(e.target.value);
            loadSlice(view, sliceIndex);
        });
    });
}

// Load all scans
async function loadScans()
{
    try
    {
        loadingState.style.display = 'block';
        scansGrid.style.display = 'none';
        emptyState.style.display = 'none';

        const response = await fetch('/api/scans');
        const data = await response.json();

        if (!response.ok)
        {
            throw new Error(data.error || 'Failed to load scans');
        }

        allScans = data.scans;
        filteredScans = [...allScans];

        displayScans(filteredScans);
        updateStats();

        loadingState.style.display = 'none';
        scansGrid.style.display = 'grid';

    } catch (error)
    {
        console.error('Error loading scans:', error);
        loadingState.style.display = 'none';
        alert('Failed to load scans: ' + error.message);
    }
}

// Filter scans based on search and format
function filterScans()
{
    const searchTerm = searchInput.value.toLowerCase();
    const formatValue = formatFilter.value;

    filteredScans = allScans.filter(scan =>
    {
        const matchesSearch = scan.name.toLowerCase().includes(searchTerm);
        const matchesFormat = formatValue === 'all' || scan.format === formatValue;
        return matchesSearch && matchesFormat;
    });

    displayScans(filteredScans);
    updateStats();
}

// Display scans in grid
function displayScans(scans)
{
    scansGrid.innerHTML = '';

    if (scans.length === 0)
    {
        scansGrid.style.display = 'none';
        emptyState.style.display = 'block';
        return;
    }

    scansGrid.style.display = 'grid';
    emptyState.style.display = 'none';

    scans.forEach(scan =>
    {
        const card = createScanCard(scan);
        scansGrid.appendChild(card);
    });
}

// Create scan card element
function createScanCard(scan)
{
    const card = document.createElement('div');
    card.className = 'scan-card';

    const iconClass = scan.format === 'nifti' ? 'fa-file-medical' : 'fa-database';

    card.innerHTML = `
        <div class="scan-card-header">
            <div class="scan-icon">
                <i class="fas ${iconClass}"></i>
            </div>
            <div class="scan-title">
                <div class="scan-name" title="${scan.name}">${scan.name}</div>
            </div>
        </div>
        <div class="scan-card-body">
            <div class="scan-detail">
                <span class="scan-detail-label">Format:</span>
                <span class="scan-detail-value">
                    <span class="scan-format">${scan.format}</span>
                </span>
            </div>
            <div class="scan-detail">
                <span class="scan-detail-label">Type:</span>
                <span class="scan-detail-value">${scan.format === 'nifti' ? 'NIfTI' : 'NumPy'}</span>
            </div>
            <div class="scan-detail">
                <span class="scan-detail-label">Extension:</span>
                <span class="scan-detail-value">${scan.format === 'nifti' ? '.nii.gz' : '.npy'}</span>
            </div>
        </div>
        <div class="scan-card-footer">
            <button class="btn-view" onclick="openViewer('${scan.name}', '${scan.format}')">
                <i class="fas fa-eye"></i> View Scan
            </button>
            <button class="btn-info" title="More info">
                <i class="fas fa-info-circle"></i>
            </button>
        </div>
    `;

    return card;
}

// Update statistics
function updateStats()
{
    totalScansEl.textContent = filteredScans.length;
}

// Open viewer modal
async function openViewer(scanName, scanFormat)
{
    currentScan = scanName;
    currentFormat = scanFormat;

    // Update modal title
    document.getElementById('modal-title').textContent = `MRI Viewer - ${scanName}`;
    document.getElementById('modal-format').textContent = scanFormat.toUpperCase();

    // Show modal
    viewerModal.style.display = 'flex';

    // Load the scan
    try
    {
        const response = await fetch('/api/load_scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                scan_name: scanName,
                format: scanFormat
            })
        });

        const data = await response.json();

        if (!response.ok)
        {
            throw new Error(data.error || 'Failed to load scan');
        }

        dimensions = data.dimensions;
        middleIndices = data.middle_indices;

        // Update UI
        document.getElementById('modal-dimensions').textContent =
            `${data.shape[0]} × ${data.shape[1]} × ${data.shape[2]}`;
        document.getElementById('scan-info-bar').style.display = 'flex';

        // Setup sliders and load middle slices
        for (var i = 0; i < views.length; i++)
        {
            const view = views[i];
            const maxSlice = dimensions[view] - 1;
            sliders[view].max = maxSlice;
            sliders[view].value = middleIndices[view];
            sliders[view].disabled = false;
            totalSpans[view].textContent = maxSlice;

            await loadSlice(view, middleIndices[view]);
        }

    } catch (error)
    {
        alert(`Error loading scan: ${error.message}`);
        console.error(error);
        closeModal();
    }
}

// Close modal
function closeModal()
{
    viewerModal.style.display = 'none';

    // Reset viewer state
    views.forEach(view =>
    {
        sliders[view].disabled = true;
        images[view].style.display = 'none';
        images[view].src = '';
        currentSpans[view].textContent = '-';
        totalSpans[view].textContent = '-';
    });

    document.getElementById('scan-info-bar').style.display = 'none';
}

// Load slice function
async function loadSlice(view, sliceIndex)
{
    try
    {
        spinners[view].style.display = 'flex';
        images[view].style.opacity = '0.5';

        const response = await fetch('/api/get_slice', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                scan_name: currentScan,
                view: view,
                slice_index: sliceIndex
            })
        });

        const data = await response.json();

        if (!response.ok)
        {
            throw new Error(data.error || 'Failed to load slice');
        }

        // Create a promise that resolves when the image loads
        await new Promise((resolve, reject) =>
        {
            const img = images[view];

            const onLoad = () =>
            {
                img.removeEventListener('load', onLoad);
                img.removeEventListener('error', onError);
                img.style.display = 'block';
                resolve();
            };

            const onError = () =>
            {
                img.removeEventListener('load', onLoad);
                img.removeEventListener('error', onError);
                reject(new Error('Failed to load image'));
            };

            img.addEventListener('load', onLoad);
            img.addEventListener('error', onError);

            // Update image source with cache busting
            img.src = data.image_url + '?t=' + new Date().getTime();
        });

        currentSpans[view].textContent = sliceIndex;
        spinners[view].style.display = 'none';
        images[view].style.opacity = '1';

    } catch (error)
    {
        console.error(`Error loading ${view} slice:`, error);
        spinners[view].style.display = 'none';
        images[view].style.opacity = '1';
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) =>
{
    if (e.key === 'Escape' && viewerModal.style.display === 'flex')
    {
        closeModal();
    }
});

// ==================== TRANSFORMATION FUNCTIONS ====================

// Transformation state
let transformationState = {
    isActive: false,
    currentView: 'axial',
    currentSlice: null,
    transformedData: null
};

// Toggle transformation panel
function toggleTransformPanel() {
    const panel = document.getElementById('transform-panel');
    const viewToggle = document.getElementById('view-toggle');
    
    if (panel.style.display === 'none') {
        panel.style.display = 'block';
        viewToggle.style.display = 'flex';
        transformationState.isActive = true;
    } else {
        panel.style.display = 'none';
        viewToggle.style.display = 'none';
        transformationState.isActive = false;
        switchView('normal');
    }
}

// Update transformation parameters based on selected type
function updateTransformParams() {
    const transformType = document.getElementById('transform-type').value;
    const paramsContainer = document.getElementById('transform-params');
    
    if (!transformType) {
        paramsContainer.innerHTML = '';
        return;
    }
    
    const params = getTransformParams(transformType);
    paramsContainer.innerHTML = generateParamsHTML(params);
    
    // Add event listeners to sliders
    paramsContainer.querySelectorAll('input[type="range"]').forEach(slider => {
        slider.addEventListener('input', (e) => {
            const valueSpan = document.getElementById(`${e.target.id}-value`);
            if (valueSpan) {
                valueSpan.textContent = e.target.value;
            }
        });
    });
}

// Get parameters for each transformation type
function getTransformParams(transformType) {
    const paramConfigs = {
        'noise_gaussian': [
            { name: 'sigma', label: 'Sigma (Noise Strength)', min: 0, max: 0.5, step: 0.01, default: 0.1 }
        ],
        'noise_salt_pepper': [
            { name: 'amount', label: 'Amount', min: 0, max: 0.2, step: 0.01, default: 0.05 }
        ],
        'noise_speckle': [
            { name: 'variance', label: 'Variance', min: 0, max: 0.3, step: 0.01, default: 0.1 }
        ],
        'filter_gaussian': [
            { name: 'sigma', label: 'Sigma (Blur Amount)', min: 0.1, max: 10, step: 0.1, default: 2.0 }
        ],
        'filter_median': [
            { name: 'size', label: 'Kernel Size', min: 3, max: 15, step: 2, default: 3 }
        ],
        'filter_sharpen': [
            { name: 'alpha', label: 'Sharpness', min: 0.5, max: 3, step: 0.1, default: 1.5 }
        ],
        'filter_edge': [
            { name: 'method', label: 'Method', type: 'select', options: ['sobel', 'canny', 'prewitt'], default: 'sobel' }
        ],
        'geometric_rotate': [
            { name: 'angle', label: 'Angle (degrees)', min: -180, max: 180, step: 5, default: 45 }
        ],
        'geometric_flip': [
            { name: 'axis', label: 'Axis', type: 'select', options: [{ value: 0, label: 'Horizontal' }, { value: 1, label: 'Vertical' }], default: 0 }
        ],
        'intensity_brightness': [
            { name: 'factor', label: 'Brightness Factor', min: 0.1, max: 3, step: 0.1, default: 1.2 }
        ],
        'intensity_contrast': [
            { name: 'factor', label: 'Contrast Factor', min: 0.1, max: 3, step: 0.1, default: 1.5 }
        ],
        'intensity_histogram': []
    };
    
    return paramConfigs[transformType] || [];
}

// Generate HTML for parameters
function generateParamsHTML(params) {
    if (params.length === 0) {
        return '<p style="grid-column: 1 / -1; text-align: center; color: var(--text-secondary);">No parameters needed</p>';
    }
    
    return params.map(param => {
        if (param.type === 'select') {
            const options = Array.isArray(param.options[0]) ? param.options : 
                           param.options.map(opt => typeof opt === 'object' ? opt : { value: opt, label: opt });
            
            return `
                <div class="param-group">
                    <label>${param.label}:</label>
                    <select id="param-${param.name}" class="control-select">
                        ${options.map(opt => `
                            <option value="${opt.value}" ${opt.value == param.default ? 'selected' : ''}>
                                ${opt.label || opt.value}
                            </option>
                        `).join('')}
                    </select>
                </div>
            `;
        } else {
            return `
                <div class="param-group">
                    <label>${param.label}:</label>
                    <input type="range" id="param-${param.name}" class="param-slider" 
                           min="${param.min}" max="${param.max}" step="${param.step}" value="${param.default}">
                    <div class="param-value" id="param-${param.name}-value">${param.default}</div>
                </div>
            `;
        }
    }).join('');
}

// Apply transformation
async function applyTransformation() {
    const transformView = document.getElementById('transform-view').value;
    const transformType = document.getElementById('transform-type').value;
    
    if (!transformType) {
        alert('Please select a transformation type');
        return;
    }
    
    const applyBtn = document.getElementById('apply-transform-btn');
    applyBtn.disabled = true;
    applyBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    
    try {
        // Get current slice index
        const sliceIndex = parseInt(sliders[transformView].value);
        
        // Parse transformation type
        const [category, subtype] = transformType.split('_');
        
        // Collect parameters
        const params = collectTransformParams(transformType);
        
        // Build API request
        const requestData = {
            scan_name: currentScan,
            view: transformView,
            slice_index: sliceIndex,
            transform_type: category,
            params: params
        };
        
        console.log('Applying transformation:', requestData);
        
        // Call API
        const response = await fetch('/api/transform', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Transformation failed');
        }
        
        // Store transformation data
        transformationState.transformedData = data;
        transformationState.currentView = transformView;
        transformationState.currentSlice = sliceIndex;
        
        // Switch to comparison view
        switchView('comparison');
        
        // Display results
        displayTransformationResults(data);
        
        console.log('Transformation successful:', data);
        
    } catch (error) {
        console.error('Transformation error:', error);
        alert(`Transformation failed: ${error.message}`);
    } finally {
        applyBtn.disabled = false;
        applyBtn.innerHTML = '<i class="fas fa-play"></i> Apply';
    }
}

// Collect parameters from UI
function collectTransformParams(transformType) {
    const params = {};
    const paramConfigs = getTransformParams(transformType);
    const [category, subtype] = transformType.split('_');
    
    // Add subtype-specific parameter
    if (category === 'add_noise') {
        params.noise_type = subtype;
    } else if (category === 'filter') {
        params.filter_type = subtype;
    } else if (category === 'geometric') {
        params.geo_type = subtype;
    } else if (category === 'intensity') {
        params.intensity_type = subtype;
    }
    
    // Collect user-defined parameters
    paramConfigs.forEach(config => {
        const element = document.getElementById(`param-${config.name}`);
        if (element) {
            params[config.name] = config.type === 'select' ? 
                                  element.value : 
                                  parseFloat(element.value);
        }
    });
    
    return params;
}

// Display transformation results
function displayTransformationResults(data) {
    // Load images
    document.getElementById('comparison-original').src = data.original_url + '?t=' + new Date().getTime();
    document.getElementById('comparison-transformed').src = data.transformed_url + '?t=' + new Date().getTime();
    document.getElementById('comparison-diff').src = data.diff_url + '?t=' + new Date().getTime();
    
    // Display metrics
    const metrics = data.metrics;
    const assessment = data.assessment;
    
    document.getElementById('metric-psnr').textContent = metrics.psnr.toFixed(2);
    document.getElementById('metric-ssim').textContent = metrics.ssim.toFixed(4);
    document.getElementById('metric-mse').textContent = metrics.mse.toFixed(6);
    document.getElementById('metric-mae').textContent = metrics.mae.toFixed(4);
    
    // Set assessments with color coding
    setAssessment('assessment-psnr', assessment.psnr);
    setAssessment('assessment-ssim', assessment.ssim);
    document.getElementById('assessment-overall').textContent = assessment.overall;
    
    // Show metrics panel
    document.getElementById('metrics-panel').style.display = 'block';
}

// Set assessment with appropriate styling
function setAssessment(elementId, text) {
    const element = document.getElementById(elementId);
    element.textContent = text;
    
    // Remove all classes
    element.className = 'metric-assessment';
    
    // Add appropriate class
    if (text.includes('Excellent')) {
        element.classList.add('excellent');
    } else if (text.includes('Good')) {
        element.classList.add('good');
    } else if (text.includes('Fair')) {
        element.classList.add('fair');
    } else if (text.includes('Poor')) {
        element.classList.add('poor');
    }
}

// Switch between normal and comparison view
function switchView(viewType) {
    const normalView = document.getElementById('normal-view');
    const comparisonView = document.getElementById('comparison-view');
    const toggleBtns = document.querySelectorAll('.toggle-btn');
    
    toggleBtns.forEach(btn => btn.classList.remove('active'));
    
    if (viewType === 'normal') {
        normalView.style.display = 'grid';
        comparisonView.style.display = 'none';
        document.querySelector('.toggle-btn:first-child').classList.add('active');
    } else {
        normalView.style.display = 'none';
        comparisonView.style.display = 'block';
        document.querySelector('.toggle-btn:last-child').classList.add('active');
    }
}

// Reset transformation
function resetTransformation() {
    // Clear transformation state
    transformationState.transformedData = null;
    
    // Reset form
    document.getElementById('transform-type').value = '';
    document.getElementById('transform-params').innerHTML = '';
    
    // Hide metrics
    document.getElementById('metrics-panel').style.display = 'none';
    
    // Switch back to normal view
    switchView('normal');
    
    console.log('Transformation reset');
}
