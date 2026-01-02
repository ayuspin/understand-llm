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
        script: "scripts/step4_softmax.py",
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
    },
    {
        title: "Step 5: Positional Encoding (Basics)",
        script: "scripts/step5_pos_encoding.py",
        explanation: `
            <p>If you process all words at the same time, the model doesn't know their order. It's "Position Blind."</p>
            <p><strong>Positional Encoding</strong> is a trick to fix this: we <strong>add</strong> a unique vector of numbers to each word vector.</p>
            <p>This "colors" the word with its location:</p>
            <ul style="margin: 16px 0; padding-left: 20px; color: var(--text-secondary);">
                <li><strong>Meaning</strong>: The original word vector.</li>
                <li><strong>Position</strong>: A unique "watermark" vector for that index.</li>
            </ul>
            <p>By adding them together, the word now carries both <strong>what</strong> it is and <strong>where</strong> it is in a single package.</p>
        `
    },
    {
        title: "Step 6: RoPE (Modern Rotation)",
        script: "scripts/step6_rope_math.py",
        explanation: `
            <p>Modern LLMs (Llama, Mistral) use <strong>RoPE</strong> (Rotary Positional Embedding) instead of simple addition.</p>
            <p>Instead of <em>adding</em> a vector, we treat word vectors as pairs of points on a 2D graph and <strong>rotate</strong> them.</p>
            <ul style="margin: 16px 0; padding-left: 20px; color: var(--text-secondary);">
                <li><strong>The Logic</strong>: Position determine the angle of rotation.</li>
                <li><strong>The Clock Handle</strong>: Each pair rotates at a different speed (frequency).</li>
                <li><strong>Relative Distance</strong>: Because rotation is consistent, the "angle" between words stays the same even if the sentence is shifted.</li>
            </ul>
            <p><strong>Run the code</strong> to see how the 'cat' vector pairs point in different directions at each position.</p>
        `
    },
    {
        title: "Step 7: Scaled Dot-Product Attention",
        script: "scripts/step7_attention_math.py",
        explanation: `
            <p>This is the "Heart" of the Transformer. Instead of fixed weights, Attention builds a custom weight matrix for every sentence.</p>
            <p>The 3-Step Process:</p>
            <ol style="margin: 16px 0; padding-left: 20px; color: var(--text-secondary);">
                <li><strong>Dot-Product ($QK^T$)</strong>: Every word's <strong>Query</strong> (Search) is compared against every word's <strong>Key</strong> (Index).</li>
                <li><strong>Scaling ($\sqrt{d}$)</strong>: Divide scores by the square root of the dimension. This keeps the math stable so Softmax doesn't "freeze."</li>
                <li><strong>Softmax & Value</strong>: Turn scores into percentages and use them to blend the <strong>Values</strong> (Actual Data).</li>
            </ol>
            <p><strong>Run the code</strong> to see how the word "sat" calculates that it should pay 85% of its attention to "cat".</p>
        `
    },
    {
        title: "Step 8: Multi-Head Attention (Parallel Experts)",
        script: "scripts/step8_multi_head_attention.py",
        explanation: `
            <p>In a real LLM, we don't just use one Attention "search." We use many in parallel—this is called <strong>Multi-Head Attention</strong>.</p>
            <ul style="margin: 16px 0; padding-left: 20px; color: var(--text-secondary);">
                <li><strong>The Projection</strong>: Instead of just slicing the vector, we use separate weight matrices to "extract" specific information for each head.</li>
                <li><strong>The Experts</strong>: Each head specializes in a different relationship (e.g., Head 1 looks for nouns, Head 2 looks for actions).</li>
                <li><strong>The Concat</strong>: We glue all the specialized results back together into one final, context-rich vector.</li>
            </ul>
            <p><strong>Run the code</strong> to see how <strong>Head 1</strong> and <strong>Head 2</strong> use their unique weight matrices to focus on completely different parts of the sentence!</p>
        `
    }
];
