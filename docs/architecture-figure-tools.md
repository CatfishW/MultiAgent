# Scientific Architecture Figure Tools — Research Notes

## Tools Evaluated

### Web-Friendly Interactive Diagrams
- **D3.js** — Full control, declarative data-driven SVG/Canvas. Best for custom interactive pipeline visualizations. Used for the dashboard architecture diagrams.
- **Mermaid.js** — Good for flowcharts, sequence diagrams, class diagrams. Limited visual polish; works well for markdown docs.
- **Cytoscape.js** — Graph theory library, excellent for network/agent graphs, automatic layouts.
- **React Flow** — Node-based editor framework, good for editable diagrams, requires React.

### Publication-Quality Static Figures
- **TikZ (LaTeX)** — Gold standard for academic papers. Vector-perfect, programmatic, but not web-friendly. Best for camera-ready paper submissions.
- **Matplotlib + patches** — Can draw any shape programmatically. Good for embedding in Python reports.
- **Graphviz (DOT)** — Automatic graph layout. Fast but rigid styling.
- **PlantUML** — Text-to-diagram. Good for quick sequence/activity diagrams.

### Animation / Video
- **Manim** — 3Blue1Brown's animation engine. Overkill for static architecture diagrams, but unmatched for explanatory animations.
- **Reveal.js + D3** — Presentation slides with embedded interactive SVGs.

### Export / Office Integration
- **python-pptx** — Programmatic PowerPoint generation. Used to export architecture diagrams as presentation slides.
- **python-docx** — Word document generation for reports.
- **ReportLab** — PDF generation with vector graphics.

## Recommendation

For the Multi-Agent dashboard:
1. **D3.js** (or vanilla SVG as implemented) for web interactive diagrams — hoverable nodes, animated edges, real-time status.
2. **python-pptx** for exportable presentation slides — one slide per architecture with labeled nodes and connectors.
3. **TikZ** as a future option if camera-ready paper figures are needed — can be generated from the same node/edge definitions.

## Implementation

- Dashboard uses pure SVG + CSS animations (no external JS library dependency).
- PPTX export endpoint available at `/export/pptx` when the web server is running.
