from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf"
OUTPUT.mkdir(parents=True, exist_ok=True)

PAGE_W, PAGE_H = A4
INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#5D687A")
BLUE = colors.HexColor("#2856A3")
TEAL = colors.HexColor("#138A8A")
PALE_BLUE = colors.HexColor("#EAF1FC")
PALE_TEAL = colors.HexColor("#E7F6F4")
PALE_GOLD = colors.HexColor("#FFF4D6")
LINE = colors.HexColor("#D8DFEA")
WHITE = colors.white


class NumberedDocTemplate(BaseDocTemplate):
    def __init__(self, filename, document_title, document_subtitle, **kwargs):
        super().__init__(filename, **kwargs)
        self.document_title = document_title
        self.document_subtitle = document_subtitle
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="content",
        )
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self._page))

    def _page(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(BLUE)
        canvas.rect(0, PAGE_H - 9 * mm, PAGE_W, 9 * mm, fill=1, stroke=0)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(WHITE)
        canvas.drawString(17 * mm, PAGE_H - 5.8 * mm, "CAUVID RESEARCH TAKEAWAYS")
        canvas.setStrokeColor(LINE)
        canvas.line(17 * mm, 13 * mm, PAGE_W - 17 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(17 * mm, 8.5 * mm, self.document_subtitle)
        canvas.drawRightString(PAGE_W - 17 * mm, 8.5 * mm, f"{doc.page}")
        canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="TitleX", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=23, leading=27, textColor=INK, alignment=TA_LEFT,
    spaceAfter=5 * mm,
))
styles.add(ParagraphStyle(
    name="Deck", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=11, leading=15, textColor=MUTED, spaceAfter=6 * mm,
))
styles.add(ParagraphStyle(
    name="H1X", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=15, leading=18, textColor=BLUE, spaceBefore=5 * mm,
    spaceAfter=2.5 * mm,
))
styles.add(ParagraphStyle(
    name="H2X", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=11.5, leading=14, textColor=INK, spaceBefore=3 * mm,
    spaceAfter=1.5 * mm,
))
styles.add(ParagraphStyle(
    name="BodyX", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.3, leading=13.1, textColor=INK, spaceAfter=2.2 * mm,
))
styles.add(ParagraphStyle(
    name="SmallX", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=7.9, leading=10.5, textColor=MUTED,
))
styles.add(ParagraphStyle(
    name="BulletX", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9, leading=12.4, leftIndent=4 * mm, firstLineIndent=-3 * mm,
    bulletIndent=0, textColor=INK, spaceAfter=1.4 * mm,
))
styles.add(ParagraphStyle(
    name="CalloutX", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=10, leading=14, textColor=INK, alignment=TA_LEFT,
))
styles.add(ParagraphStyle(
    name="FormulaX", parent=styles["BodyText"], fontName="Courier-Bold",
    fontSize=8.2, leading=11, textColor=BLUE, alignment=TA_CENTER,
))


def p(text, style="BodyX"):
    return Paragraph(text, styles[style])


def bullet(text):
    return Paragraph(f"- {text}", styles["BulletX"])


def callout(text, color=PALE_BLUE):
    table = Table([[p(text, "CalloutX")]], colWidths=[170 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("BOX", (0, 0), (-1, -1), 0.6, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2 * mm),
    ]))
    return table


def section(title, body):
    return KeepTogether([p(title, "H2X"), *body])


def make_doc(path, title, subtitle, story, top=19 * mm, bottom=18 * mm):
    doc = NumberedDocTemplate(
        str(path), title, subtitle,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=top,
        bottomMargin=bottom,
        title=title,
        author="CauVid Research Notes",
        subject=subtitle,
    )
    doc.build(story)


def executive_summary():
    story = [
        p("From Video to Defensible Interpretation", "TitleX"),
        p(
            "Executive takeaways on annotation-free ego-motion segmentation, "
            "neuro-symbolic validation, and evaluation.",
            "Deck",
        ),
        callout(
            "Core position: CauVid is an evidence-grounded neuro-symbolic video "
            "reasoning system. It does not claim an unavailable absolute annotation; "
            "it constructs competing interpretations and tests which one provides the "
            "most coherent, falsifiable account of the observed video."
        ),
        Spacer(1, 3 * mm),
    ]

    rows = [
        [p("Pipeline through Step 6", "H2X"), p("Step 7: latent segmentation", "H2X")],
        [
            p(
                "Select videos; detect and label objects; associate detections into "
                "persistent tracks; then combine 2D boxes with per-frame depth maps "
                "to estimate camera-frame 3D positions. Former Steps 4-5 are bypassed."
            ),
            p(
                "Estimate lateral and longitudinal ego-motion signals, generate "
                "multiple symbolic segmentations, stabilize short interruptions, "
                "identify parameter regions with invariant outputs, and pass plausible "
                "candidates downstream rather than committing prematurely."
            ),
        ],
    ]
    table = Table(rows, colWidths=[84 * mm, 84 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), PALE_TEAL),
        ("BACKGROUND", (1, 0), (1, 0), PALE_GOLD),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
    ]))
    story.extend([table, Spacer(1, 3 * mm)])

    story.extend([
        section("Why multiple candidates matter", [
            bullet("They increase hypothesis coverage and reduce irreversible errors from choosing one boundary too early."),
            bullet("A stable parameter plateau is an empirical robustness margin: nearby parameter changes leave the segmentation unchanged."),
            bullet("Stability is not correctness. Stable but degenerate or biased explanations must be challenged downstream."),
        ]),
        section("How validation works without annotations", [
            bullet("Proposal evidence generates candidates; independent validation evidence tests them."),
            bullet("Depth, background flow, object-relative motion, temporal continuity, physical constraints, and defeasible traffic knowledge provide support or contradiction."),
            bullet("Every conclusion becomes an evidence record: claim, support, contradictions, alternatives, uncertainty, and provenance."),
        ]),
        section("Strongest research contribution", [
            p(
                "The central contribution is the epistemic architecture - set-valued "
                "hypothesis generation, independent cross-stage validation, defeasible "
                "symbolic reasoning, auditable provenance, and closed-loop revision - "
                "rather than the threshold mechanism alone."
            ),
        ]),
        section("Primary evaluation bundle", [
            bullet("Evidence-grounded self-consistency, reported as components rather than only one scalar."),
            bullet("Explanatory coverage and abstention; severity-weighted contradiction rate."),
            bullet("Independence-adjusted cross-modal agreement and perturbation robustness."),
            bullet("Held-out downstream and predictive utility; explanation faithfulness under intervention."),
            bullet("Candidate diversity, redundancy, effective candidate count, expert audit, and computational cost."),
        ]),
        Spacer(1, 2 * mm),
        callout(
            "Defensible claim: the system identifies interpretations that are "
            "comprehensive, stable, non-circular, falsifiable, and useful downstream. "
            "It does not claim absolute segmentation accuracy without annotations.",
            PALE_TEAL,
        ),
    ])
    return story


def detailed_note():
    story = [
        p("Evidence-Grounded Neuro-Symbolic Video Reasoning", "TitleX"),
        p(
            "Research takeaways for latent ego-motion segmentation, downstream "
            "validation, explainability, and annotation-free evaluation.",
            "Deck",
        ),
        callout(
            "Research thesis: maintain multiple plausible interpretations of the "
            "video, expose the evidence behind each one, test them with independent "
            "signals and defeasible knowledge, preserve ambiguity when warranted, "
            "and revise upstream beliefs when downstream contradictions emerge."
        ),
        p("1. Pipeline context", "H1X"),
        p(
            "Steps 1-6 establish the perceptual substrate. The pipeline selects the "
            "dataset scope, performs per-frame object detection and labeling, links "
            "detections into persistent tracks, and estimates camera-frame 3D object "
            "positions by combining 2D bounding boxes with depth maps. Former Steps "
            "4-5 are removed, so the depth-enriched tracking state directly supplies "
            "the inputs needed for later motion reasoning. Invalid or incomplete depth "
            "artifacts are regeneration targets rather than trusted cache entries."
        ),
        p("2. What Step 7 is doing", "H1X"),
        p(
            "Step 7 treats ego-motion segmentation as latent inference. Smoothed "
            "lateral and longitudinal motion signals do not admit one universally "
            "reliable discretization because their scale and noise vary by video. "
            "The system therefore creates a family of candidate symbolic sequences - "
            "left/static/right and backward/static/forward - and applies temporal "
            "regularization to suppress brief interruptions and bridge compatible "
            "segments. Candidates are retained as inputs to downstream reasoning."
        ),
        p(
            "The map from a decision parameter to its symbolic sequence is piecewise "
            "constant. A contiguous parameter interval that produces the same sequence "
            "is therefore a local invariance region. Its width measures tolerance to "
            "parameter perturbation and is useful as an empirical robustness margin. "
            "This idea is related to stability-based model selection and persistence, "
            "but plateau width alone is not a correctness certificate."
        ),
        callout(
            "Stability means: the interpretation does not depend on one finely tuned "
            "parameter. It does not mean: the interpretation is semantically true.",
            PALE_GOLD,
        ),
        p("3. Why multiple candidates help", "H1X"),
        p(
            "A set-valued intermediate representation postpones an irreversible "
            "choice. Adding candidates weakly improves oracle coverage in theory, "
            "because the best member of a larger set cannot be worse than the best "
            "member of its subset. In this dataset, however, the oracle segmentation "
            "is unavailable; the pipeline must not present this mathematical fact as "
            "measured recall. The practical claim is instead that multiple candidates "
            "increase hypothesis coverage and give downstream evidence an opportunity "
            "to resolve ambiguity."
        ),
        p("Candidate coverage x selection reliability -> final interpretation quality", "FormulaX"),
        p(
            "The first factor is improved by diverse proposals. The second depends on "
            "whether later stages can reject stable but incorrect, redundant, or "
            "degenerate hypotheses. High candidate coverage without reliable selection "
            "simply transfers uncertainty downstream."
        ),
        p("4. The system being developed", "H1X"),
        p(
            "The appropriate positioning is an evidence-grounded, neuro-symbolic, "
            "abductive video reasoning system. Neural perception supplies uncertain "
            "measurements: objects, labels, tracks, depth, velocities, and spatial "
            "relations. Symbolic knowledge supplies interpretable concepts and "
            "constraints: object semantics, physical compatibility, temporal logic, "
            "traffic conventions, and causal expectations. The reasoning layer asks "
            "which structured interpretation best explains the observations while "
            "minimizing contradiction and unnecessary complexity."
        ),
        p("H* = argmax [evidence + prediction + knowledge - contradiction - complexity]", "FormulaX"),
        p(
            "Symbolic knowledge should be defeasible. Physical impossibilities may be "
            "encoded as hard constraints, but traffic conventions and typical behavior "
            "should usually be soft constraints because real road users can violate "
            "them. Strong visual evidence must be able to override a prior rule, and "
            "unresolved conflicts must remain explicit rather than being forced into a "
            "predefined explanation."
        ),
        p("5. The strongest contribution", "H1X"),
        bullet("Set-valued perception: retain plausible alternatives instead of hiding uncertainty in hard predictions."),
        bullet("Independent validation: separate proposal evidence from evidence used to support or falsify a hypothesis."),
        bullet("Evidence ledger: attach support, contradictions, alternatives, uncertainty, and provenance to every symbolic claim."),
        bullet("Defeasible neuro-symbolic inference: combine perceptual evidence with physical and semantic knowledge without treating priors as infallible."),
        bullet("Closed-loop revision: use downstream contradictions to revisit upstream candidates rather than merely lowering final confidence."),
        p(
            "This epistemic architecture is more general and promising than the "
            "specific threshold mechanism. The threshold sweep is one proposal engine; "
            "the main scientific question is how the complete system knows what it "
            "knows, identifies what could falsify a claim, and revises beliefs when new "
            "evidence conflicts with them."
        ),
        p("6. Evidence-grounded self-consistency", "H1X"),
        p(
            "Self-consistency should be a primary metric, but logical agreement among "
            "the system's own outputs is insufficient. An empty explanation, a shared "
            "upstream error, or a forced rule-compliant narrative can all appear "
            "consistent. The metric must reward explanatory coverage and independent "
            "evidence while penalizing contradictions."
        ),
        p(
            "EGSC = Coverage x [logic + temporal + cross-modal + predictive + robustness] x exp(-contradictions)",
            "FormulaX",
        ),
        p(
            "Report the normalized components as a vector as well as an aggregate. "
            "Evaluate consistency at claim, segment, and video levels. Cross-modal "
            "support should be discounted when two apparently distinct signals share "
            "the same provenance, preventing circular confirmation."
        ),
        p("7. Complementary evaluation metrics", "H1X"),
    ]

    metric_rows = [
        [p("Dimension", "H2X"), p("Recommended measures", "H2X"), p("Failure it detects", "H2X")],
        [p("Coverage", "SmallX"), p("Explained frames, tracks, events; abstention rate", "SmallX"), p("Trivially sparse explanations", "SmallX")],
        [p("Contradictions", "SmallX"), p("Severity-weighted violations; unresolved conflicts", "SmallX"), p("Coherent-looking but incompatible claims", "SmallX")],
        [p("Cross-modal", "SmallX"), p("Independence-adjusted agreement among depth, flow, tracks, and motion", "SmallX"), p("Circular confirmation", "SmallX")],
        [p("Robustness", "SmallX"), p("Claim overlap, boundary displacement, catastrophic-change rate", "SmallX"), p("Sensitivity to noise or tuning", "SmallX")],
        [p("Downstream utility", "SmallX"), p("Held-out prediction, trajectory consistency, contradiction reduction", "SmallX"), p("Internally plausible but useless representations", "SmallX")],
        [p("Faithfulness", "SmallX"), p("Evidence deletion, counterfactual response, provenance completeness", "SmallX"), p("Post-hoc explanations", "SmallX")],
        [p("Candidate quality", "SmallX"), p("Diversity, redundancy, effective count, discrimination, ambiguity retention", "SmallX"), p("Many candidates without useful alternatives", "SmallX")],
        [p("Parsimony", "SmallX"), p("Segments, claims, graph size, explanation efficiency", "SmallX"), p("Overly elaborate narratives", "SmallX")],
        [p("External audit", "SmallX"), p("Expert support judgments, missed contradictions, inter-rater agreement", "SmallX"), p("Misalignment of automatic metrics", "SmallX")],
        [p("Efficiency", "SmallX"), p("Runtime, memory, candidate pruning, cost per explained event", "SmallX"), p("Impractical inference", "SmallX")],
    ]
    metrics = Table(metric_rows, colWidths=[31 * mm, 87 * mm, 50 * mm], repeatRows=1)
    metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#F7F9FC")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.2 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.2 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8 * mm),
    ]))
    story.extend([
        metrics,
        p("8. Experimental design", "H1X"),
        p(
            "Use held-out videos and compare: no Step 7 representation; a fixed "
            "single segmentation; one locally selected candidate; multiple candidates "
            "without validation; and the full cross-stage validation system. Perform "
            "ablations for plateau stability, symbolic rules, evidence independence, "
            "contradiction propagation, and candidate diversity. Perturb depth, boxes, "
            "tracks, frames, thresholds, and random seeds at multiple magnitudes."
        ),
        p(
            "A small blinded expert audit is valuable even when dense annotations are "
            "unavailable. Experts can judge whether cited evidence supports a claim, "
            "whether contradictions were missed, whether alternatives are appropriate, "
            "and whether confidence reflects ambiguity. The key validation is whether "
            "higher automatic self-consistency predicts stronger downstream utility, "
            "perturbation robustness, and expert-rated plausibility."
        ),
        p("9. Claims to make - and avoid", "H1X"),
        callout(
            "Make: the method constructs comprehensive, stable, non-circular, "
            "falsifiable, and downstream-useful interpretations from unannotated video.",
            PALE_TEAL,
        ),
        Spacer(1, 2.5 * mm),
        callout(
            "Avoid: claims of absolute segmentation correctness, measured recall, or "
            "ground-truth recovery unless an external annotation or sensor reference is "
            "introduced.",
            PALE_GOLD,
        ),
        p("10. Candidate ICLR contribution statement", "H1X"),
        p(
            "We introduce an annotation-free, evidence-grounded neuro-symbolic "
            "framework for video understanding that maintains multiple latent "
            "interpretations and validates them through independent perceptual "
            "evidence, defeasible symbolic knowledge, and downstream consistency. "
            "Rather than treating explanations as post-hoc descriptions, the framework "
            "represents each conclusion as an auditable hypothesis with explicit "
            "support, contradictions, alternatives, uncertainty, and provenance. "
            "Downstream evidence can revise uncertain upstream interpretations, "
            "yielding a closed-loop reasoning process evaluated through evidence-grounded "
            "self-consistency, explanatory coverage, robustness, faithfulness, and "
            "held-out utility."
        ),
        p("References and conceptual anchors", "H1X"),
        p(
            "Meinshausen and Buhlmann (2010), Stability Selection, JRSS-B. "
            "Cohen-Steiner, Edelsbrunner, and Harer (2007), Stability of Persistence "
            "Diagrams, Discrete & Computational Geometry. Ben-David, von Luxburg, "
            "and Pal (2006), A Sober Look at Clustering Stability. These works support "
            "the robustness motivation while also underscoring that stability alone is "
            "not semantic correctness.",
            "SmallX",
        ),
    ])
    return story


if __name__ == "__main__":
    make_doc(
        OUTPUT / "cauvid_takeaways_executive_summary.pdf",
        "From Video to Defensible Interpretation",
        "Executive summary",
        executive_summary(),
        top=17 * mm,
        bottom=17 * mm,
    )
    make_doc(
        OUTPUT / "cauvid_neurosymbolic_research_takeaways.pdf",
        "Evidence-Grounded Neuro-Symbolic Video Reasoning",
        "Detailed research note",
        detailed_note(),
        top=27 * mm,
    )
    print(OUTPUT / "cauvid_takeaways_executive_summary.pdf")
    print(OUTPUT / "cauvid_neurosymbolic_research_takeaways.pdf")
