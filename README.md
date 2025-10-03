<h1>Chest X-ray Pneumonia Classification — CNN (HW3)</h1>

<p>
This notebook builds and evaluates Convolutional Neural Networks (CNNs) to classify pediatric chest X-ray images
as <em>NORMAL</em> vs <em>PNEUMONIA</em>. It covers dataset loading/visualization, a compact custom CNN, a ResNet-18
head fine-tuning setup, fast CPU training, and validation-set evaluation.
</p>

<hr/>

<h2>📁 Project Layout</h2>
<pre><code>.
├─ HW3_CNN.ipynb
├─ HW3_CNN-lib/
│  ├─ data/
│  │  ├─ train/
│  │  │  ├─ NORMAL/...
│  │  │  └─ PNEUMONIA/...
│  │  └─ val/
│  │     ├─ NORMAL/...
│  │     └─ PNEUMONIA/...
│  └─ resnet18_weights_9.pth
└─ README.md
</code></pre>

<hr/>

<h2>🧠 Problem &amp; Data</h2>
<ul>
  <li><strong>Goal:</strong> Binary classify AP chest X-rays (pediatric, 1–5 years) into NORMAL vs PNEUMONIA.</li>
  <li><strong>Counts (provided split):</strong> train → NORMAL=335, PNEUMONIA=387; val → NORMAL=64, PNEUMONIA=104.</li>
  <li><strong>Paths:</strong> <code>DATA_PATH = "../HW3_CNN-lib/data"</code>, <code>WEIGHT_PATH = "../HW3_CNN-lib/resnet18_weights_9.pth"</code>.</li>
  <li><strong>Reproducibility:</strong> all seeds set to <code>24</code> (Python/NumPy/Torch) and <code>PYTHONHASHSEED="24"</code>.</li>
</ul>

<hr/>

<h2>🛠️ Environment</h2>
<pre><code>python &gt;= 3.9
pip install torch torchvision numpy scikit-learn matplotlib
</code></pre>

<hr/>

<h2>🚦 Pipeline</h2>

<h3>1) Load &amp; Visualize (20 pts)</h3>
<ul>
  <li><code>get_count_metrics(folder)</code> returns per-class counts for <code>train</code>/<code>val</code>/<code>test</code>.</li>
  <li><code>load_data()</code> uses <code>torchvision.datasets.ImageFolder</code> with transforms:
    <ul>
      <li><code>transforms.RandomResizedCrop(224)</code></li>
      <li><code>transforms.ToTensor()</code></li>
    </ul>
    <em>batch_size=32</em>, <em>shuffle=True</em> (train), <em>shuffle=False</em> (val). On the provided split, <code>len(train_loader)==23</code>.
  </li>
  <li>Utility to display mini-batches for sanity checks.</li>
</ul>

<h3>2) Build Models (35 pts)</h3>

<h4>2.1 Convolution Output Volume (10 pts)</h4>
<p>
Warm-up helper: for input size <code>W</code>, filter <code>F</code>, stride <code>S</code>, padding <code>P</code>, output size is
<code>floor((W - F + 2P)/S) + 1</code>.
</p>

<h4>2.2 Custom <code>SimpleCNN</code> (15 pts)</h4>
<ul>
  <li>Backbone: 3 conv blocks (Conv3×3 → ReLU → MaxPool2d(2,2)) producing <code>128×28×28</code>.</li>
  <li>Head: Flatten → Dropout(0.5) → Linear(100352→256) → ReLU → Linear(256→2).</li>
  <li>Constraints satisfied: <em>&lt;20 modules</em>, final logits shape <code>(B,2)</code>, size ≈ <em>0.103 GB</em> of parameters.</li>
</ul>

<h4>2.3 Predefined CNN (ResNet-18) (10 pts)</h4>
<ul>
  <li>Instantiate <code>torchvision.models.resnet18(pretrained=False)</code>.</li>
  <li>Replace <code>model.fc</code> with <code>Linear(in_features, 2)</code> for binary logits.</li>
  <li><em>Freeze</em> all parameters except the new <code>fc</code> layer; load weights from <code>resnet18_weights_9.pth</code>.</li>
</ul>

<h3>3) Train (25 pts)</h3>
<ul>
  <li><strong>Loss:</strong> <code>nn.CrossEntropyLoss()</code> (raw logits).</li>
  <li><strong>Optimizer:</strong> <code>torch.optim.SGD(model.parameters(), lr=1e-2)</code>.</li>
  <li><strong>Epochs:</strong> <code>n_epochs=1</code> (CPU-friendly).</li>
  <li><strong>Loop:</strong> zero_grad → forward → loss → backward → step; print mean epoch loss.</li>
</ul>

<h3>4) Evaluate (20 pts)</h3>
<ul>
  <li><code>eval_model()</code> returns <em>predicted labels</em> (argmax) and ground-truth labels.</li>
  <li>Metric: validation accuracy via <code>sklearn.metrics.accuracy_score</code>.</li>
</ul>

<hr/>

<h2>📈 Observed Results (final run)</h2>
<ul>
  <li><strong>Epoch 0</strong> mean training loss: <code>0.182608</code>.</li>
  <li><strong>Validation Accuracy:</strong> <code>0.8690476190</code> (N=168).</li>
  <li><strong>Total runtime (CPU):</strong> ~<code>426.78 s</code>.</li>
</ul>

<hr/>

<h2>📊 Reproducibility Notes</h2>
<ul>
  <li>Seed everything to 24 (Python/NumPy/PyTorch) and set <code>PYTHONHASHSEED</code>.</li>
  <li>Use <code>model.eval()</code> and <code>torch.no_grad()</code> for evaluation.</li>
  <li>Keep transforms identical across runs (<code>RandomResizedCrop(224)</code>, <code>ToTensor()</code>).</li>
</ul>

<hr/>

<h2>⚠️ Challenges &amp; How They Were Solved</h2>
<ol>
  <li>
    <strong>Accuracy threshold (&gt; 0.7) on validation</strong><br/>
    <em>Approach:</em> Fine-tuned a ResNet-18 head (binary <code>fc</code>) while freezing the backbone; used CrossEntropyLoss with SGD (lr=1e-2). Achieved ~0.869 accuracy on the val set in one epoch.
  </li>
  <li>
    <strong>Output format mismatch for evaluation</strong><br/>
    <em>Issue:</em> Using probabilities for <code>accuracy_score</code> causes errors/mismatch.<br/>
    <em>Fix:</em> Return integer class labels via <code>argmax(logits, dim=1)</code>, and keep ground truth as 1-D int arrays.
  </li>
  <li>
    <strong>Model/parameter limits for the custom CNN</strong><br/>
    <em>Constraint:</em> &lt;20 modules and ≤1 GB model size; output shape must be <code>(B,2)</code>.<br/>
    <em>Fix:</em> Designed a 3-block CNN with a small MLP head (≈0.103 GB), validating module count and output shape by asserts.
  </li>
  <li>
    <strong>CPU runtime budget</strong><br/>
    <em>Observation:</em> End-to-end run took ~426.8s on CPU, above the nominal 4-minute target.<br/>
    <em>Mitigations:</em> Kept <code>n_epochs=1</code> and froze the ResNet backbone. If a stricter CPU budget is required, consider:
    <ul>
      <li>Setting <code>num_workers</code> &gt; 0 and <code>pin_memory=True</code> on DataLoaders.</li>
      <li>Reducing visualization calls and I/O during runs.</li>
      <li>Switching to a smaller crop size (e.g., 192) consistently across train/val to cut compute.</li>
      <li>Optionally training/evaluating only the head or using <code>torch.compile</code> (if available).</li>
    </ul>
  </li>
</ol>

<hr/>

<h2>🚀 How to Run</h2>
<ol>
  <li>Open <code>HW3_CNN.ipynb</code>.</li>
  <li>Run cells in order: setup → counts/visualization → models → training → evaluation.</li>
  <li>Confirm <em>val accuracy &gt; 0.7</em> and final runtime completes on your machine.</li>
</ol>

<hr/>

<h2>📄 License</h2>
<p>MIT .</p>

<hr/>

<h2>🙌 Acknowledgements</h2>
<p>Thanks to the course staff for the dataset split, pretrained weights, and assignment scaffolding.</p>
