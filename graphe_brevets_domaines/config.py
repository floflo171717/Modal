"""Constantes du graphe brevets par domaines d'abstract."""
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
GRAPH_PATH = DATA / "graphs" / "patent_domain_graph.graphml"
SUMMARY_PATH = DATA / "stats" / "summary.json"
DOMAINS_PATH = DATA / "stats" / "domains.json"
SYNONYM_PATH = DATA / "stats" / "synonyms.json"

N_SAMPLE = 50
SEED = 42
SPACY_MODEL = "en_core_web_sm"
MIN_SHARED = 2

PATENT_NOISE = {
    "present invention", "the invention", "the present invention", "the disclosure",
    "one embodiment", "some embodiments", "certain embodiments", "an embodiment",
    "the embodiment", "said embodiment", "this embodiment", "another embodiment",
    "the method", "the system", "the device", "the apparatus", "the composition",
    "the processor", "the memory", "the computer", "the user", "the network",
    "the present disclosure", "the following description", "the accompanying drawings",
    "a method", "a system", "a device", "a computer", "a processor", "a memory",
    "a plurality", "one or more", "at least one", "the at least one", "at least a",
    "first aspect", "second aspect", "third aspect", "fourth aspect",
    "technical field", "background art", "brief description", "detailed description",
    "method", "system", "device", "apparatus", "module", "unit", "element",
    "component", "portion", "structure", "process", "step", "means",
    "data", "information", "signal", "input", "output", "result", "value",
    "example", "aspect", "feature", "use", "application", "field",
    "improved performance", "industrial deployment", "embodiments provide",
    "this document", "that document", "the document", "these constraints", "both these",
    "other users", "another user", "the user", "each motion", "motion event",
    "one or more processors", "one or more modules", "one or more devices",
    "one or more units", "one or more components", "one or more elements",
    "one or more steps", "one or more processes", "one or more systems",
    "at least one processor", "at least one module", "at least one device",
    "a plurality of", "plurality of",
    "action items", "product definitions", "computing resources", "common attributes",
    "event processing", "control code", "analytic code", "archival resources",
    "alternative resources", "application instances", "available computing",
    "automatically generated", "automatically-generated",
    # bruit observe sur echantillon 100 abstracts (seed=42)
    "energy consumption", "health care medical image presentation",
    "locate contractors", "proprietary or industry standard protocol",
    "software containers", "surgical hardware", "technology domains",
    "industry standard protocol", "standard protocol", "image presentation",
    "medical image presentation", "decision support system",
}

# Mots génériques brevets : si tous les tokens d'un terme y sont -> bruit.
PATENT_NOISE_TOKENS = {
    "access", "action", "alternative", "alternatives", "anonymous", "apparatus", "application",
    "applications", "archival", "attribute", "attributes", "automation", "available", "average",
    "board", "candidate", "certain", "code", "common", "component", "components", "composition",
    "computer", "computing", "constraint", "constraints", "content", "continuous", "control",
    "data", "definition", "definitions", "device", "devices", "document", "documents", "element",
    "elements", "embodiment", "event", "events", "example", "feature", "features", "field",
    "information", "input", "instance", "instances", "interest", "invention", "item", "items",
    "means", "memory", "method", "module", "modules", "motion", "network", "object", "objects",
    "output", "patent", "patents", "portion", "process", "processes", "processor", "processors",
    "product", "products", "query", "queries", "resource", "resources", "result", "signal",
    "slide", "slides", "step", "steps", "sticker", "stickers", "structure", "subpart", "system",
    "unit", "units", "use", "user", "users", "value", "view", "point", "each", "node", "nodes",
    "capability", "capabilities", "platform", "contractors", "containers", "markers",
    "presentation", "consumption", "protocol", "domains", "locate",
    "proprietary", "standard", "surgical",
}

# Débuts typiques de noun chunks boilerplate (determinants / pronoms brevets).
PATENT_NOISE_PREFIXES = (
    "the ", "a ", "an ", "said ", "such ", "this ", "that ", "these ", "those ",
    "each ", "every ", "any ", "all ", "both ", "other ", "another ", "one ", "at least ",
    "one or more ", "a plurality of ", "plurality of ",
    "nearly all ", "nearly ", "reduced ", "definable ",
)

# Indices qu'un terme ressemble à un domaine scientifique/technique (pour noun chunks hors seed).
DOMAIN_SIGNALS = (
    "science", "sciences", "engineering", "technology", "technologies", "computing",
    "industry", "medicine", "medical", "biology", "biological", "chemistry", "chemical",
    "physics", "learning", "intelligence", "informatics", "robotics", "biomedical", "clinical",
    "environmental", "electrical", "mechanical", "software", "hardware", "telecommunication",
    "telecommunications", "semiconductor", "genetics", "genomic", "oncology", "pharmaceutical",
    "blockchain", "reality", "recognition", "imaging", "wireless", "optical", "quantum",
    "nanotechnology", "cybersecurity", "automotive", "aerospace", "agriculture", "energy",
    "sustainability", "manufacturing", "logistics", "finance", "economics", "psychology",
    "neuroscience", "linguistics", "geology", "astronomy", "meteorology", "hydrology",
    "medicine", "health", "healthcare", "legal", "law", "contract", "contracts",
)

MANUAL_SYNONYMS: dict[str, list[str]] = {
    "artificial intelligence": ["ai", "a.i.", "ai-based"],
    "machine learning": ["ml", "machine-learning"],
    "natural language processing": ["nlp", "natural-language processing"],
    "information technology": ["it", "information technologies"],
    "medicine": ["healthcare", "health care", "health-care", "medical", "clinical", "biomedical", "medical imaging"],
    "automotive industry": ["automotive", "automobile industry", "vehicle industry"],
    "computer science": ["computing", "computer technology"],
    "environmental science": ["environmental sciences", "environmental technology"],
    "materials science": ["material science", "materials engineering"],
    "political science": ["politics"],
    "food science": ["food sciences", "food technology"],
    "electrical engineering": ["electrical and electronic engineering"],
    "mechanical engineering": ["mechanical systems engineering"],
    "biomedical engineering": ["bioengineering", "biomedical technology"],
    "cloud computing": ["cloud-based computing", "cloud services"],
    "wireless communication": ["wireless communications", "wireless networking"],
    "image processing": ["image analysis", "digital image processing"],
    "speech recognition": ["automatic speech recognition", "voice recognition"],
    "drug delivery": ["pharmaceutical delivery", "medication delivery"],
    "renewable energy": ["clean energy", "green energy"],
    "augmented reality": ["augmented reality markers", "ar", "mixed reality"],
    "blockchain": ["blockchain smart contracts", "smart contracts", "distributed ledger"],
    "clinical decision support": ["clinical decision support systems", "clinical decision-support", "cds"],
}

DOMAIN_PATTERNS: list[tuple[str, list[dict], int]] = [
    ("field_of", [{"LOWER": "in"}, {"LOWER": "the"}, {"LOWER": "field"}, {"LOWER": "of"}, {"POS": {"IN": ["DET", "ADJ", "NOUN", "PROPN"]}, "OP": "+"}], 4),
    ("fields_of", [{"LOWER": {"IN": ["field", "fields"]}}, {"LOWER": "of"}, {"POS": {"IN": ["DET", "ADJ", "NOUN", "PROPN"]}, "OP": "+"}], 2),
    ("for_use_in", [{"LOWER": "for"}, {"LOWER": "use"}, {"LOWER": "in"}, {"POS": {"IN": ["DET", "ADJ", "NOUN", "PROPN"]}, "OP": "+"}], 3),
    ("application_in", [{"LOWER": {"IN": ["application", "applications"]}}, {"LOWER": {"IN": ["in", "of", "for", "to"]}}, {"POS": {"IN": ["DET", "ADJ", "NOUN", "PROPN"]}, "OP": "+"}], 2),
    ("applied_to", [{"LOWER": "applied"}, {"LOWER": "to"}, {"POS": {"IN": ["DET", "ADJ", "NOUN", "PROPN"]}, "OP": "+"}], 2),
    ("industry", [{"POS": {"IN": ["ADJ", "NOUN", "PROPN"]}, "OP": "+"}, {"LOWER": "industry"}], 0),
]
