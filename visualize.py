"""
Visualize epistemic deformation manifold across variants

Generates visualization plots showing how different stressors (confidence pressure,
resource constraints, reframing) affect epistemic integrity metrics.

For full suite results, run: python cli.py suite "your prompt" --output-dir ./results
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from harness import compare_topology

# Configuration: Change these paths to match your results location
# Use "." when running validate.py (results_*.json in current directory)
# Use "./results" when using CLI suite output (python cli.py suite ... --output-dir ./results)
RESULTS_DIR = "."  # Current directory for results_*.json files

# Load all variant results
# Try to load from results_*.json files first (from validate.py)
# then fall back to full_suite_*.json format
variants = ["baseline", "commitment_pressure", "metaphor", "adversarial", 
            "literal", "confidence", "token_200", "token_100"]

data = {}
for v in variants:
    # Try results_*.json format first (from validate.py)
    result_path = os.path.join(RESULTS_DIR, f"results_{v}.json")
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                data[v] = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {result_path}: {e}")
    else:
        # Fall back to full_suite_*.json format
        full_suite_path = os.path.join(RESULTS_DIR, f"full_suite_{v}.json")
        if os.path.exists(full_suite_path):
            try:
                with open(full_suite_path) as f:
                    data[v] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {full_suite_path}: {e}")
        else:
            print(f"Warning: {v} not found at {result_path} or {full_suite_path}, skipping")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Epistemic Stress Harness - Deformation Manifold', fontsize=16, fontweight='bold')

# Extract metrics
names = list(data.keys())
commitment_latency = [data[v]["metrics"]["commitment_latency"] for v in names]
branch_count = [data[v]["metrics"]["branch_count"] for v in names]
tokens_per_cp = [data[v]["metrics"]["tokens_per_checkpoint"] for v in names]
claim_select = [data[v]["metrics"]["claim_select_ratio"] for v in names]

# Color code by stressor type
colors = []
labels = []
for v in names:
    if v == "baseline":
        colors.append('#2c3e50')
        labels.append('Baseline')
    elif 'metaphor' in v or 'literal' in v or 'adversarial' in v:
        colors.append('#27ae60')
        labels.append('Reframing')
    elif 'confidence' in v or 'commitment' in v:
        colors.append('#e74c3c')
        labels.append('Incentive')
    elif 'token' in v:
        colors.append('#3498db')
        labels.append('Resource')
    else:
        colors.append('#95a5a6')
        labels.append('Other')

# Plot 1: Commitment Latency
ax1 = axes[0, 0]
bars1 = ax1.bar(range(len(names)), commitment_latency, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Commitment Latency', fontsize=11, fontweight='bold')
ax1.set_title('When Does Commitment Occur?', fontsize=12)
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels([n.replace('_', '\n') for n in names], rotation=45, ha='right', fontsize=9)
ax1.axhline(y=commitment_latency[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')
ax1.set_ylim(0, 0.5)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Branch Preservation
ax2 = axes[0, 1]
bars2 = ax2.bar(range(len(names)), branch_count, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Branch Count', fontsize=11, fontweight='bold')
ax2.set_title('Exploration vs Assertion', fontsize=12)
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels([n.replace('_', '\n') for n in names], rotation=45, ha='right', fontsize=9)
ax2.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='Baseline')
ax2.set_ylim(0, 3)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Justification Density
ax3 = axes[1, 0]
bars3 = ax3.bar(range(len(names)), tokens_per_cp, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Tokens per Checkpoint', fontsize=11, fontweight='bold')
ax3.set_title('Justification Density', fontsize=12)
ax3.set_xticks(range(len(names)))
ax3.set_xticklabels([n.replace('_', '\n') for n in names], rotation=45, ha='right', fontsize=9)
ax3.axhline(y=tokens_per_cp[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Scatter - Density vs Commitment
ax4 = axes[1, 1]
scatter = ax4.scatter(commitment_latency, tokens_per_cp, c=colors, s=200, alpha=0.7, 
                     edgecolor='black', linewidth=1.5)

# Annotate points
for i, name in enumerate(names):
    ax4.annotate(name.replace('_', '\n'), 
                (commitment_latency[i], tokens_per_cp[i]),
                fontsize=8, ha='center', va='bottom', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

ax4.set_xlabel('Commitment Latency', fontsize=11, fontweight='bold')
ax4.set_ylabel('Tokens per Checkpoint', fontsize=11, fontweight='bold')
ax4.set_title('Response Geometry', fontsize=12)
ax4.grid(True, alpha=0.3)

# Add quadrant lines
ax4.axvline(x=0.30, color='gray', linestyle='--', alpha=0.3)
ax4.axhline(y=35, color='gray', linestyle='--', alpha=0.3)

# Create legend
legend_elements = [
    mpatches.Patch(facecolor='#2c3e50', edgecolor='black', label='Baseline'),
    mpatches.Patch(facecolor='#27ae60', edgecolor='black', label='Reframing'),
    mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Incentive Pressure'),
    mpatches.Patch(facecolor='#3498db', edgecolor='black', label='Resource Constraint')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
          ncol=4, frameon=False, fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_dir = os.path.join(RESULTS_DIR, "visualizations")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "deformation_manifold.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: {output_path}")

# Create second figure: Topology distance
fig2, ax = plt.subplots(figsize=(12, 6))
fig2.suptitle('Topology Distance from Baseline', fontsize=14, fontweight='bold')

# Skip baseline itself
variant_names = names[1:]
variant_colors = colors[1:]

# Compute real topology comparisons vs baseline
# If baseline data is not available, skip topology visualization
if "baseline" in data and "checkpoints" in data["baseline"]:
    baseline_checkpoints = data["baseline"]["checkpoints"]
    node_overlap_data = []
    
    for variant_name in variant_names:
        if variant_name in data and "checkpoints" in data[variant_name]:
            variant_checkpoints = data[variant_name]["checkpoints"]
            topology = compare_topology(baseline_checkpoints, variant_checkpoints)
            node_overlap_data.append(topology.node_overlap)
        else:
            # If variant data not available, use 0.0 as placeholder
            print(f"Warning: No checkpoint data for {variant_name}, defaulting to 0.0 (missing data)")
            node_overlap_data.append(0.0)
    
    # Ensure we have data for all variants
    if len(node_overlap_data) != len(variant_names):
        print(f"ERROR: Topology data length mismatch. Expected {len(variant_names)}, got {len(node_overlap_data)}")
        # Pad with 0.0 if needed
        node_overlap_data = (node_overlap_data + [0.0] * len(variant_names))[:len(variant_names)]
else:
    print("Warning: Baseline checkpoint data not available, topology visualization will use fallback values (0.0)")
    node_overlap_data = [0.0] * len(variant_names)

x = range(len(variant_names))
width = 0.7

bars = ax.bar(x, node_overlap_data, width, color=variant_colors, alpha=0.7, 
             edgecolor='black', linewidth=1.5)

ax.set_ylabel('Node Overlap (Jaccard Similarity)', fontsize=11, fontweight='bold')
ax.set_title('Structural Preservation Under Stress', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([n.replace('_', '\n') for n in variant_names], rotation=45, ha='right')
ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Frame Invariant (>95%)')
ax.axhline(y=0.80, color='orange', linestyle='--', alpha=0.5, label='Moderate Drift')
ax.axhline(y=0.60, color='red', linestyle='--', alpha=0.5, label='Topology Collapse')
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)
ax.legend(loc='lower right')

plt.tight_layout()
output_path2 = os.path.join(output_dir, "topology_distance.png")
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: {output_path2}")

plt.close('all')
print(f"\nVisualization complete. Check {output_dir}/ directory for outputs.")
