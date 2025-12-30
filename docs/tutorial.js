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
function loadStep(index) {
    if (index < 0 || index >= STEPS.length) return;

    currentStep = index;
    const step = STEPS[index];

    titleEl.textContent = step.title;
    explanationEl.innerHTML = step.explanation;
    codeEditor.value = step.code;
    outputEl.textContent = '👆 Click "Run" to execute this code';

    // Update navigation
    stepCounter.textContent = `Step ${index + 1} of ${STEPS.length}`;
    prevBtn.disabled = index === 0;
    nextBtn.disabled = index === STEPS.length - 1;

    // Update progress bar
    const progress = ((index + 1) / STEPS.length) * 100;
    progressBar.style.width = `${progress}%`;
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
