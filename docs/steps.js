// Tutorial Steps Configuration
// Code is dynamically loaded from docs/scripts/*.py files
const STEPS = [
    {
        title: "Step 1: Matrix Multiplication (x @ W)",
        script: "scripts/step1_matrix_math.py",
        explanation: `
            <p><strong>The Foundation: Multiplying a Row by a Matrix.</strong></p>
            <p>We take a horizontal <strong>Row Vector</strong> and multiply it against the columns of a <strong>Matrix</strong>.</p>
            <ul style="margin: 16px 0; padding-left: 20px; color: var(--text-secondary);">
                <li><strong>The Input (x)</strong>: A vector of size 3 (e.g., [2, 5, 1]).</li>
                <li><strong>The Matrix (W)</strong>: A 3x2 grid of numbers.</li>
                <li><strong>The Operation (Dot Product)</strong>: Matrix multiplication consists of calculating the <strong>Dot Product</strong> between the Row of the first matrix and each Column of the second. This means pairing up the numbers, multiplying them, and adding the results together.</li>
            </ul>
            <p><strong>Run the code</strong> to see how the single Input Row is used in two separate Dot Product operations (one for each column) to produce the final result.</p>
        `
    },
    {
        title: "Step 2: Softmax (Probability)",
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
        title: "Step 3: Positional Encoding (Basics)",
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
        title: "Step 4: RoPE (Modern Rotation)",
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
        title: "Step 5: Scaled Dot-Product Attention",
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
        title: "Step 6: Multi-Head Attention (Parallel Experts)",
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
    },
    {
        title: "Step 7: Layer Normalization (The Stabilizer)",
        script: "scripts/step9_layer_norm.py",
        explanation: `
            <p>Deep networks have a problem: as numbers travel through many layers, they can become <strong>explosively large</strong> or <strong>microscopically small</strong>. This makes training impossible.</p>
            <ul style="margin: 16px 0; padding-left: 20px; color: var(--text-secondary);">
                <li><strong>The Mean (The Center)</strong>: We find the average value of all features in a word. If the output is shifted too far positive or negative, we pull it back to 0.</li>
                <li><strong>The Variance (The Spread)</strong>: we measure how much the values "pop" or "scatter." We scale them so the spread is always 1, preventing numerical explosions.</li>
                <li><strong>Learned Tuning</strong>: The model can still choose to make certain features "louder" using parameters called Gamma and Beta.</li>
            </ul>
            <p><strong>Run the code</strong> to see how a messy, unstable vector is instantly tamed and re-centered!</p>
        `
    }
];
