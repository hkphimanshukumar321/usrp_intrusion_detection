import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.axis('off')

    # Draw boxes
    def draw_box(ax, x, y, width, height, text, facecolor='#E8F0FE'):
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='black', facecolor=facecolor)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, horizontalalignment='center', verticalalignment='center', fontsize=11, fontweight='bold', wrap=True)

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))

    # Input
    draw_box(ax, 0.1, 0.90, 0.8, 0.08, "Input Spectrogram [B, 3, 224, 224]\n(Generated via STFT)", facecolor='#FFF3E0')
    draw_arrow(ax, 0.5, 0.90, 0.5, 0.83)

    # Multi-Scale Backbone
    draw_box(ax, 0.1, 0.70, 0.8, 0.13, "ResNet-50 Multi-Scale Backbone\n(Timm Pre-trained ImageNet Extractor)\nOutputs: Stage 2 (512ch) | Stage 3 (1024ch) | Stage 4 (2048ch)", facecolor='#E3F2FD')
    draw_arrow(ax, 0.5, 0.70, 0.5, 0.63)

    # Projection
    draw_box(ax, 0.1, 0.55, 0.8, 0.08, "1x1 Convolution Projections\nAligning all stages to common 512-channel dimension", facecolor='#FCE4EC')
    draw_arrow(ax, 0.5, 0.55, 0.5, 0.48)

    # ASPP + Coordinate Attention 
    draw_box(ax, 0.05, 0.20, 0.9, 0.28, "", facecolor='#F3E5F5')
    ax.text(0.5, 0.45, "Feature Enrichment & Coordinate Attention", horizontalalignment='center', fontsize=13, fontweight='bold')
    
    draw_box(ax, 0.1, 0.35, 0.35, 0.08, "LightASPP (Stage 4 only)\n(Multi-scale dilated convolutions)", facecolor='#E1BEE7')
    draw_box(ax, 0.55, 0.35, 0.35, 0.08, "Coordinate Attention (All Stages)\nFactorized Pooling (Time Strip x Freq Strip)", facecolor='#E1BEE7')
    
    draw_arrow(ax, 0.45, 0.39, 0.55, 0.39)
    draw_arrow(ax, 0.5, 0.20, 0.5, 0.13)

    # Classifier
    draw_box(ax, 0.1, 0.03, 0.8, 0.1, "AdaptiveAvgPool2D + Concat (1536 dims)\n-> Dense(512) -> BN -> ReLU -> Dropout -> Dense(256)", facecolor='#E8F5E9')
    draw_arrow(ax, 0.5, 0.03, 0.5, -0.04)
    
    # Output
    draw_box(ax, 0.1, -0.12, 0.8, 0.08, "Logits Output [B, 6 Classes]\nOptimized via Focal Loss (\u03b3=2.0)", facecolor='#FFEB3B')

    plt.title("SDR_Proposed_CoordASPP_Focal Architecture (IEEE Schematic)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('architecture_diagram_CoordASPP.png', dpi=300, bbox_inches='tight')
    print("Saved architecture diagram to architecture_diagram_CoordASPP.png")

if __name__ == "__main__":
    draw_architecture()
