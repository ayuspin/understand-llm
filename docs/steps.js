// Tutorial Steps Configuration
// Code is dynamically loaded from docs/scripts/*.py files
const STEPS = [
    {
        title: "Step 1: The Dot Product",
        script: "scripts/step1_dot_product.py",
        explanation: `
            <p>The <strong>Dot Product</strong> is the core building block of all neural network math.</p>
            <p>It's surprisingly simple: <strong>multiply pairs of numbers together, then add up the results</strong>.</p>
            <p>In this example, we have two lists:</p>
            <p><code>list_a = [10, 20]</code> (our input data)</p>
            <p><code>list_b = [1, 2]</code> (our weights)</p>
            <p>The calculation is: <code>(10 × 1) + (20 × 2) = 10 + 40 = 50</code></p>
            <p>This single number tells us how much the input "aligns" with the weights. Higher alignment = higher score.</p>
            <p><strong>Click "Run" to see it in action! →</strong></p>
        `
    },
    {
        title: "Step 2: Matrix Multiplication",
        script: "scripts/step2_matrix_multiply.py",
        explanation: `
            <p>A <strong>Matrix</strong> is just a stack of weight vectors.</p>
            <p>When we multiply a matrix by a vector, we're doing a <strong>dot product for every row</strong>.</p>
            <p>In AI terminology:</p>
            <ul style="margin: 16px 0; padding-left: 20px; color: var(--text-secondary);">
                <li>Each <strong>row</strong> in the matrix is a "detector" for a specific feature.</li>
                <li>The result is a new vector of scores—one for each detector.</li>
            </ul>
            <p>This is how neural networks "compare" inputs against learned patterns.</p>
        `
    },
    {
        title: "Step 3: The AI Orientation (x × W)",
        script: "scripts/step3_ai_orientation.py",
        explanation: `
            <p>In real AI code, we flip the order: <strong>Input × Weights</strong> instead of Weights × Input.</p>
            <p>This means:</p>
            <ul style="margin: 16px 0; padding-left: 20px; color: var(--text-secondary);">
                <li>Input is a <strong>horizontal row</strong></li>
                <li>Each <strong>column</strong> in the weight matrix is a word's weights</li>
            </ul>
            <p>This is exactly how language models work. They take a hidden state (row) and multiply it against an output matrix where each column represents a different word in the vocabulary.</p>
            <p>The highest score wins—that's the predicted word!</p>
        `
    },
    {
        title: "Step 4: Softmax (Probability)",
        script: "scripts/softmax_basics.py",
        explanation: `
            <p>Raw scores (Logits) are hard to compare. <strong>Softmax</strong> turns them into percentages.</p>
            <p>The 3-step recipe:</p>
            <ol style="margin: 16px 0; padding-left: 20px; color: var(--text-secondary);">
                <li><strong>Exponentiate (e^x)</strong>: Makes everything positive and amplifies the winner.</li>
                <li><strong>Sum</strong>: Total up all the scores.</li>
                <li><strong>Normalize</strong>: Divide each score by the total.</li>
            </ol>
            <p>The result is a neat "Probability Budget" that adds up to exactly 1.0 (100%).</p>
        `
    }
];
