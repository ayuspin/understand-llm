// Tutorial Controller
let pyodide = null;
let currentStep = 0;

// DOM Elements
const titleEl = document.getElementById('step-title');
const explanationEl = document.getElementById('step-explanation');
const codeEditor = document.getElementById('code-editor');
const outputEl = document.getElementById('output');
const runBtn = document.getElementById('run-btn');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const stepCounter = document.getElementById('step-counter');
const progressBar = document.getElementById('progress');

// Initialize Pyodide
async function initPyodide() {
    outputEl.textContent = '⏳ Loading Python environment...';
    try {
        pyodide = await loadPyodide();
        await pyodide.loadPackage('numpy');
        runBtn.disabled = false;
        outputEl.textContent = '✅ Python ready! Click "Run" to execute.';
    } catch (error) {
        outputEl.textContent = '❌ Failed to load Python: ' + error.message;
    }
}

// Fetch Python code from file
async function fetchCode(scriptPath) {
    try {
        const response = await fetch(scriptPath);
        if (!response.ok) throw new Error(`Failed to load ${scriptPath}`);
        return await response.text();
    } catch (error) {
        return `# Error loading script: ${error.message}`;
    }
}

// Run Python Code
async function runCode() {
    if (!pyodide) return;

    const code = codeEditor.value;
    outputEl.textContent = '⏳ Running...';

    try {
        // Redirect stdout
        pyodide.runPython(`
import sys
from io import StringIO
sys.stdout = StringIO()
        `);

        // Run user code
        await pyodide.runPythonAsync(code);

        // Get stdout
        const stdout = pyodide.runPython('sys.stdout.getvalue()');
        outputEl.textContent = stdout || '(No output)';
    } catch (error) {
        outputEl.textContent = '❌ Error:\n' + error.message;
        outputEl.style.color = '#f85149';
        setTimeout(() => outputEl.style.color = '', 2000);
    }
}

// Load Step
async function loadStep(index) {
    if (index < 0 || index >= STEPS.length) return;

    currentStep = index;
    const step = STEPS[index];

    titleEl.textContent = step.title;
    explanationEl.innerHTML = step.explanation;

    // Fetch the Python code from the script file
    codeEditor.value = '# Loading...';
    const code = await fetchCode(step.script);
    codeEditor.value = code;

    outputEl.textContent = '👆 Click "Run" to execute this code';

    // Update navigation
    stepCounter.textContent = `Step ${index + 1} of ${STEPS.length}`;
    prevBtn.disabled = index === 0;
    nextBtn.disabled = index === STEPS.length - 1;

    // Update progress bar
    const progress = ((index + 1) / STEPS.length) * 100;
    progressBar.style.width = `${progress}%`;

    // Render step navigation links
    renderStepNav();
}

// Render Step Navigation
function renderStepNav() {
    const navEl = document.getElementById('step-nav');
    navEl.innerHTML = '<div class="step-nav-title">All Steps</div>';

    STEPS.forEach((step, i) => {
        const link = document.createElement('div');
        link.className = 'step-link' + (i === currentStep ? ' active' : '');
        link.textContent = step.title;
        link.onclick = () => loadStep(i);
        navEl.appendChild(link);
    });
}

// Event Listeners
runBtn.addEventListener('click', runCode);
prevBtn.addEventListener('click', () => loadStep(currentStep - 1));
nextBtn.addEventListener('click', () => loadStep(currentStep + 1));

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        runCode();
    }
});

// Initialize
initPyodide();
loadStep(0);

// ===== RESIZABLE OUTPUT PANEL =====
const resizer = document.getElementById('resizer');
const outputSection = document.getElementById('output-section');

let isResizing = false;
let startY = 0;
let startHeight = 0;

resizer.addEventListener('mousedown', (e) => {
    isResizing = true;
    startY = e.clientY;
    startHeight = outputSection.offsetHeight;
    resizer.classList.add('dragging');
    document.body.style.cursor = 'ns-resize';
    document.body.style.userSelect = 'none';
});

document.addEventListener('mousemove', (e) => {
    if (!isResizing) return;

    // Calculate new height (dragging up = bigger output)
    const delta = startY - e.clientY;
    const newHeight = Math.max(60, Math.min(600, startHeight + delta));
    outputSection.style.height = newHeight + 'px';
});

document.addEventListener('mouseup', () => {
    if (isResizing) {
        isResizing = false;
        resizer.classList.remove('dragging');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
    }
});
