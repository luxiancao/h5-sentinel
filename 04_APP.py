import streamlit as st
import pandas as pd
import numpy as np
import joblib
from Bio import SeqIO
from Bio.Seq import Seq
import io
import re
import plotly.graph_objects as go
import os

# ================= 1. Page Configuration =================
st.set_page_config(page_title="H5 Sentinel: Mammalian Adaptation Risk Assessment", layout="wide")

T_LOW = 0.63
T_HIGH = 0.79

# ================= 2. Feature Extraction Rules (Machine Learning Track) =================
AA_FEATURES = {
    'A': [-0.591, -1.302, -0.733, 1.57, -0.146], 'C': [-1.343, 0.465, -0.862, -1.02, -0.255],
    'D': [1.05, 0.302, 3.656, 0.945, -0.371], 'E': [1.357, -1.453, 1.477, 0.567, -0.177],
    'F': [-1.006, -0.590, 1.891, -1.46, 0.412], 'G': [-0.384, 1.652, 1.330, 1.044, 0.189],
    'H': [0.336, -0.417, -0.544, 0.567, 0.023], 'I': [-1.239, -0.547, -1.337, -0.861, 0.304],
    'K': [1.831, -0.561, 0.533, -0.277, -0.296], 'L': [-1.019, -0.987, -1.505, -0.511, 0.212],
    'M': [-0.663, -1.524, 0.279, -0.911, 0.195], 'N': [0.945, 0.828, 1.299, -0.179, 0.105],
    'P': [0.189, 2.081, -1.628, 0.421, 0.282], 'Q': [0.931, -0.179, -0.425, 0.044, -0.091],
    'R': [1.538, -0.055, 1.502, 0.440, -0.375], 'S': [-0.228, 0.421, -0.591, 0.660, 0.188],
    'T': [-0.032, 0.666, -1.333, 0.325, 0.291], 'V': [-1.337, -0.279, -1.622, -0.912, 0.243],
    'W': [-0.595, 0.009, 0.672, -2.128, 0.675], 'Y': [0.260, 0.830, 0.259, 0.051, 0.296],
    '-': [-0.13, -0.117, 0.6025, -0.028, -0.112], '?': [0, 0, 0, 0, 0], '*': [0, 0, 0, 0, 0]
}
DEFAULT_VECTOR = [0.0] * 5

# Preserved strictly for ML Feature Matrix alignment
GENE_RULES = {
    'HA': {
        'Single': [110, 112, 123, 125, 128, 132, 137, 139, 142, 144, 149, 150, 155, 156, 157, 158, 168, 170, 171, 172, 181, 184, 187, 198, 199, 202, 204, 205, 208, 209, 211, 212, 217, 220, 224, 226, 228, 229, 237, 238, 239, 240, 251, 257, 258, 267, 347, 403, 410, 468, 469],
        'Combo': ['91K+139P+205K+513K', '119Y+172A+239L+240S', '123R+124I'], 'Deletion': [],
        'JSD': {131: (0.196, 'Q'), 226: (0.030, 'A'), 526: (0.029, 'V'), 120: (0.024, 'M')}
    },
    'NA': {
        'Single': [46, 142, 224, 287, 319, 325, 392, 430], 'Combo': [],
        'Deletion': ['49-68', '54-72', '54-73', '62-64', '57-65'],
        'JSD': {67: (0.316, 'I'), 339: (0.210, 'P'), 269: (0.196, 'M'), 321: (0.182, 'I')}
    },
    'PB2': {
        'Single': [9, 25, 63, 158, 192, 199, 251, 253, 256, 283, 271, 274, 292, 339, 358, 389, 473, 478, 482, 483, 526, 535, 588, 591, 598, 627, 631, 645, 701, 714],
        'Combo': ['147L+627K', '89V+309D+339K+477G+495V+627E+676T', '293V+398K+588V+598M+648V+676M', '147T+339T+588T'], 'Deletion': [],
        'JSD': {362: (0.290, 'G'), 631: (0.285, 'L'), 109: (0.281, 'I'), 495: (0.279, 'I'), 676: (0.273, 'A'), 441: (0.250, 'N'), 649: (0.223, 'I'), 139: (0.176, 'I'), 478: (0.047, 'I')}
    },
    'PB1': {
        'Single': [3, 105, 577, 622, 677, 678], 'Combo': [], 'Deletion': [],
        'JSD': {430: (0.129, 'K'), 179: (0.128, 'I'), 375: (0.100, 'S'), 694: (0.098, 'N'), 59: (0.078, 'S')}
    },
    'PA': {
        'Single': [37, 49, 63, 97, 100, 142, 158, 195, 206, 210, 347, 356, 367, 383, 409, 421, 433, 444, 479, 615],
        'Combo': ['343S+347E', '142R+147V+171V+182L', '44I+127A+241Y+343T+573V', '149P+266R+357K+515T', '311I+343S', '32T+550L', '404S+409N', '49Y+347G', '314L+342Q'], 'Deletion': [],
        'JSD': {219: (0.289, 'I'), 558: (0.205, 'L'), 277: (0.201, 'P'), 113: (0.149, 'R'), 441: (0.044, 'V'), 608: (0.043, 'S'), 85: (0.033, 'A'), 61: (0.029, 'M')}
    },
    'NP': {
        'Single': [41, 52, 105, 109, 184, 210, 227, 229, 313, 319, 357, 434, 450, 470, 475],
        'Combo': ['31K+450G', '288V+437M'], 'Deletion': [],
        'JSD': {482: (0.234, 'N'), 450: (0.144, 'N'), 52: (0.082, 'H'), 105: (0.026, 'M')}
    },
    'M': {
        'Single': [30, 37, 43, 95, 215], 'Combo': [], 'Deletion': [],
        'JSD': {227: (0.150, 'T'), 200: (0.034, 'V')}
    },
    'NS': {
        'Single': [9, 42, 74, 92, 106, 138, 139, 149, 207],
        'Combo': ['103F+106M', '3S+41K+74N', '55E+66E+138F'], 'Deletion': ['80-84', '222-230'],
        'JSD': {87: (0.080, 'P'), 139: (0.053, 'N')}
    }
}

# ================= 3. Literature Detection & Formatting Track =================
LIT_MUTATIONS = {
    'HA': [
        ('110', 'N'), ('112', 'N'), ('123', 'R'), ('125', 'D'), ('128', 'R'), ('132', 'P'),
        ('137', 'N'), ('139', 'P'), ('142', '-'), ('144', ['-', 'A']), ('149', ['A', 'M']),
        ('150', ['V', 'A']), ('155', 'R'), ('156', ['S', 'E']), ('157', ['G', 'N']),
        ('158', 'S'), ('168', 'R'), ('170', ['N', '-']), ('171', 'N'), ('172', ['A', 'D', 'N', 'V']),
        ('181', '-'), ('184', 'Q'), ('187', 'N'), ('198', ['K', 'D', 'N', 'I']), ('199', ['G', 'P']),
        ('202', ['G', 'V']), ('204', 'I'), ('205', ['R', 'T', 'D']), ('208', ['R', 'H']),
        ('209', 'K'), ('211', 'I'), ('212', 'N'), ('217', 'A'), ('220', 'E'), ('224', 'I'),
        ('226', 'I'), ('228', 'L'), ('229', 'Q'), ('237', ['D', 'G']), ('238', 'L'),
        ('239', ['N', 'L']), ('240', ['A', 'S']), ('251', 'S'), ('257', 'I'), ('258', 'K'),
        ('267', 'K'), ('347', 'I'), ('403', 'I'), ('410', 'E'), ('468', 'T'), ('469', 'R'),
        ('combo', '91K+139P+205K+513K'), ('combo', '119Y+172A+239L+240S'), ('combo', '123R+124I'),
        ('120', 'M')
    ],
    'NA': [
        ('deletion', '49-68'), ('deletion', '54-72'), ('deletion', '54-73'), ('deletion', '62-64'), ('deletion', '57-65'),
        ('46', 'D'), ('142', 'E'), ('224', 'M'), ('287', 'N'), ('319', 'F'), ('325', 'S'), ('392', 'D'), ('430', 'G'),
        ('67', 'I')
    ],
    'PB2': [
        ('9', 'N'), ('25', 'A'), ('63', 'T'), ('158', ['G', 'K']), ('192', 'K'), ('199', 'S'),
        ('251', 'K'), ('253', 'N'), ('256', 'G'), ('283', 'L'), ('271', 'A'), ('274', 'T'),
        ('292', 'V'), ('339', 'T'), ('358', 'V'), ('389', 'R'), ('473', 'T'), ('478', 'I'),
        ('482', 'R'), ('483', 'K'), ('526', 'R'), ('535', 'L'), ('588', 'V'), ('591', ['K', 'R']),
        ('598', ['T', 'I']), ('627', ['K', 'V']), ('631', 'L'), ('645', 'I'), ('701', ['V', 'N']),
        ('714', 'R'),
        ('combo', '147L+627K'), ('combo', '89V+309D+339K+477G+495V+627E+676T'),
        ('combo', '292V+389K+588V+598M+648V+676M'), ('combo', '147T+339T+588T')
    ],
    'PB1': [
        ('3', 'V'), ('105', 'S'), ('577', 'E'), ('622', 'G'), ('677', 'M'), ('678', 'N'),
        ('375', 'N'), ('430', 'K'), ('694', 'N')
    ],
    'PA': [
        ('37', 'A'), ('49', 'Y'), ('63', 'I'), ('97', 'I'), ('100', 'V'), ('142', ['N', 'E']),
        ('158', 'R'), ('195', 'K'), ('206', 'R'), ('210', 'L'), ('347', 'G'), ('356', 'R'),
        ('367', 'K'), ('383', 'D'), ('409', 'S'), ('421', 'I'), ('433', 'K'), ('444', 'D'),
        ('479', 'R'), ('615', 'N'),
        ('combo', '343S+347E'), ('combo', '142R+147V+171V+182L'), ('combo', '44I+127A+241Y+343T+573V'),
        ('combo', '149P+266R+357K+515T'), ('combo', '311I+343S'), ('combo', '32T+550L'),
        ('combo', '404S+409N'), ('combo', '49Y+347G'), ('combo', '314L+342Q'),
        ('61', 'M'), ('441', 'V')
    ],
    'NP': [
        ('41', 'V'), ('52', ['H', 'N']), ('105', 'V'), ('109', 'T'), ('184', 'K'), ('210', 'D'),
        ('227', 'R'), ('229', 'R'), ('319', 'K'), ('434', 'K'), ('470', 'R'), ('357', ['L', 'K']),
        ('475', 'V'), ('313', ['Y', 'V']), ('450', 'N'), ('482', 'S'),
        ('combo', '99K+345N'), ('combo', '31K+450G'), ('combo', '286V+437M')
    ],
    'M': [
        ('30', 'D'), ('37', 'A'), ('43', 'M'), ('95', 'K'), ('215', 'A')
    ],
    'NS': [
        ('9', 'Y'), ('42', 'S'), ('74', 'N'), ('92', 'E'), ('106', 'M'), ('138', 'F'),
        ('139', 'D'), ('149', 'A'), ('207', 'D'),
        ('deletion', '80-84'), ('deletion', '222-230'),
        ('combo', '103F+106M'), ('combo', '3S+41K+74N'), ('combo', '55E+66E+138F')
    ]
}

KEY_TARGETS = [
    ('HA', '120', 'M'), ('HA', '211', 'I'), ('HA', '226', 'I'),
    ('NA', '46', 'D'), ('NA', '67', 'I'),
    ('PA', '61', 'M'), ('PA', '441', 'V'),
    ('PB2', '292', 'V'), ('PB2', '627', 'K'),
    ('PB1', '375', 'N'), ('PB1', '430', 'K'), ('PB1', '694', 'N'),
    ('NP', '52', 'H'), ('NP', '105', 'V'), ('NP', '450', 'N'), ('NP', '482', 'S'),
    ('PB2', 'combo', '89V+309D+339K+477G+495V+627E+676T')
]

def format_mut_display(gene, mut_str):
    """Maps recognized mutations to their H3/N2 standard numbering format."""
    if gene == 'HA':
        mapping = {
            '110N': 'D101N', '112N': 'D103N', '123R': 'S114R', '125D': '116D', '128R': 'S119R',
            '132P': 'S123P', '137N': 'S126N', '139P': 'S128P', 'Del 142': 'Del 131',
            'Del 144': 'Del 133', '144A': 'S133A', '149A': 'S137A', '149M': 'R137M',
            '150V': 'A138V', '150A': 'S138A', '155R': 'G143R', '156S': 'N144S', '156E': 'N144E',
            '157G': 'D145G', '157N': 'D145N', '158S': 'G146S', '168R': 'Q156R',
            '170N': 'S158N', 'Del 170': 'Del 158', '171N': 'S159N', '172A': 'T160A',
            '172D': 'A160D', '172N': 'A160N', '172V': 'A160V', 'Del 181': 'Del 169',
            '184Q': 'R172Q', '187N': 'S175N', '198K': 'N186K', '198D': 'N186D',
            '198N': 'V186N', '198I': 'V186I', '199G': 'D187G', '199P': 'T187P',
            '202G': 'E190G', '202V': 'E190V', '204I': 'T192I', '205R': 'K193R', '205T': 'K193T',
            '205D': 'N193D', '208R': 'Q196R', '208H': 'Q196H', '209K': 'N197K', '211I': 'T199I',
            '212N': 'D200N', '217A': 'T205A', '220E': 'D208E', '224I': 'T212I', '226I': 'V214I',
            '228L': 'V216L', '229Q': 'L217Q', '237D': 'G225D', '237G': 'W225G', '238L': 'Q226L',
            '239N': 'S227N', '239L': 'M227L', '240A': 'G228A', '240S': 'G228S', '251S': 'P239S',
            '257I': 'V245I', '258K': 'R246K', '267K': 'E255K', '347I': 'L331I', '403I': 'K387I',
            '410E': 'K394E', '468T': 'A452T', '469R': 'G453R',
            '91K+139P+205K+513K': 'E83K,S128P,N197K,R496K',
            '119Y+172A+239L+240S': 'H110Y,T160A,Q226L,G228S',
            '123R+124I': 'S114R,T115I',
            '120M': '109M'
        }
        h3 = mapping.get(mut_str, "")
        if h3: return f"{gene}_{mut_str} (H3 Numbering: {h3})"
    elif gene == 'NA':
        mapping = {
            '49-68 deletion': '49-68 deletion', '54-72 deletion': '54-72 deletion',
            '54-73 deletion': '54-73 deletion', '62-64 deletion': '62-64 deletion',
            '57-65 deletion': '57-65 deletion', '46D': '46D', '142E': '142E',
            '224M': '224M', '287N': '272N', '319F': '319F', '325S': '322S',
            '392D': '389D', '430G': '430G', '67I': '67I'
        }
        n2 = mapping.get(mut_str, "")
        if n2: return f"{gene}_{mut_str} (N2 Numbering: {n2})"
    return f"{gene}_{mut_str}"

# ================= 4. Dynamic Model Loading Paths =================
MODEL_PATHS = {
    "Full-length (8 Segments)": {
        "model": "model_optimization_baseline/best_model_CatBoost.pkl",
        "features": "cache_results/boruta_cols_Knowledge-Guided_Signatures.pkl"
    },
    "HA Only": {
        "model": "cache_results/best_model_CatBoost_HA.pkl",
        "features": "cache_results/features_HA.pkl"
    },
    "PB2 Only": {
        "model": "cache_results/best_model_CatBoost_PB2.pkl",
        "features": "cache_results/features_PB2.pkl"
    }
}

# ================= 5. Utility Functions =================
def clean_sequence(seq_str):
    seq = seq_str.strip().upper()
    if seq.endswith('*'): seq = seq[:-1]
    return seq

def check_combination(seq, combo_str):
    conditions = combo_str.split('+')
    match_count = 0
    for cond in conditions:
        match = re.match(r"(\d+)([A-Z])", cond)
        if match:
            pos = int(match.group(1)) - 1
            if pos < len(seq) and seq[pos] == match.group(2):
                match_count += 1
    return 1 if match_count == len(conditions) else 0

def check_deletion(seq, del_str):
    match = re.match(r"(\d+)-(\d+)", del_str)
    if match:
        start, end = int(match.group(1)) - 1, int(match.group(2))
        if start >= len(seq): return 0
        end = min(end, len(seq))
        region = seq[start:end]
        if len(region) > 0 and ((region.count('-') + region.count('?')) / len(region)) > 0.8:
            return 1
    return 0

@st.cache_resource
def load_models(segment_option):
    paths = MODEL_PATHS.get(segment_option)
    try:
        model = joblib.load(paths["model"])
        sel_cols = joblib.load(paths["features"])
        return model, sel_cols
    except Exception as e:
        st.error(f"Error loading model resources for {segment_option}. Please ensure the paths exist. Details: {e}")
        return None, None

def process_fasta_multi(fasta_text, is_nucleotide):
    strains_dict = {}
    
    fasta_text = re.sub(r'^([A-Za-z0-9]+\s*\|)', r'>\1', fasta_text, flags=re.MULTILINE)
    fasta_io = io.StringIO(fasta_text)
    
    GENE_MAPPING = {
        'PB2': 'PB2', 'PB1': 'PB1', 'PA': 'PA', 'HA': 'HA', 'NP': 'NP',
        'NA': 'NA', 'M1': 'M', 'M': 'M', 'NS1': 'NS', 'NS': 'NS'
    }
    
    for record in SeqIO.parse(fasta_io, "fasta"):
        header = record.description.upper()
        
        if '|' in header:
            gene_part = header.split('|')[0]
            strain_id = header.split('|')[1].strip()
        else:
            gene_part = header
            strain_id = record.id.strip()
            
        match = re.search(r'\b(PB2|PB1|PA|HA|NP|NA|M1|M|NS1|NS)\b', gene_part)
        if not match:
            match = re.search(r'\b(PB2|PB1|PA|HA|NP|NA|M1|M|NS1|NS)\b', header)
            
        if match:
            detected_gene = GENE_MAPPING[match.group(1)]
        else:
            continue 
            
        raw_seq_str = str(record.seq).upper()

        if is_nucleotide:
            protein_seq = ""
            for i in range(0, len(raw_seq_str) - len(raw_seq_str)%3, 3):
                codon = raw_seq_str[i:i+3]
                
                if "-" in codon or "." in codon or "~" in codon:
                    if codon.count("-") + codon.count(".") + codon.count("~") == 3:
                        protein_seq += "-"
                    else:
                        protein_seq += "X" 
                else:
                    try:
                        protein_seq += str(Seq(codon).translate())
                    except:
                        protein_seq += "X"
            seq_to_save = protein_seq
        else:
            seq_to_save = raw_seq_str
            
        if strain_id not in strains_dict:
            strains_dict[strain_id] = {}
            
        strains_dict[strain_id][detected_gene] = clean_sequence(seq_to_save)
        
    return strains_dict

def extract_features(seq_dict, sel_cols):
    features = {col: 0.0 for col in sel_cols}
    key_muts = []
    other_muts = []

    # 1. Feature Extraction for ML Model compatibility
    for gene, rules in GENE_RULES.items():
        if gene not in seq_dict: continue
        seq = seq_dict[gene]
        
        lit_sites = set(rules['Single'])
        jsd_sites = set(rules['JSD'].keys())
        all_physico_sites = sorted(list(lit_sites | jsd_sites))
        
        for site in all_physico_sites:
            idx = site - 1
            aa = seq[idx] if idx < len(seq) else '-'
            props = AA_FEATURES.get(aa, DEFAULT_VECTOR)
            for i, val in enumerate(props):
                col_name = f'{gene}_{site}_Dim{i+1}'
                if col_name in features:
                    features[col_name] = val
                    
        for site, (score, target_aa) in rules['JSD'].items():
            idx = site - 1
            col_name = f'{gene}_JSDScore_{site}'
            if col_name in features:
                features[col_name] = score if (idx < len(seq) and seq[idx] == target_aa) else 0.0
                
        if 'Combo' in rules:
            for combo in rules['Combo']:
                col_name = f'{gene}_Combo_{combo}'
                is_present = check_combination(seq, combo)
                if col_name in features:
                    features[col_name] = is_present
                    
        for deletion in rules['Deletion']:
            col_name = f'{gene}_Del_{deletion}'
            is_deleted = check_deletion(seq, deletion)
            if col_name in features:
                features[col_name] = is_deleted

    # 2. UI Detection Track: Key vs Other Literature Mutations
    for gene, rules in LIT_MUTATIONS.items():
        if gene not in seq_dict: continue
        seq = seq_dict[gene]
        for rule in rules:
            mut_type = rule[0]
            if mut_type == 'combo':
                combo_str = rule[1]
                if check_combination(seq, combo_str):
                    fmt_str = format_mut_display(gene, combo_str)
                    if (gene, 'combo', combo_str) in KEY_TARGETS:
                        key_muts.append(fmt_str)
                    else:
                        other_muts.append(fmt_str)
            elif mut_type == 'deletion':
                del_str = rule[1]
                if check_deletion(seq, del_str):
                    fmt_str = format_mut_display(gene, f"{del_str} deletion")
                    other_muts.append(fmt_str)
            else:
                pos = int(rule[0])
                targets = rule[1] if isinstance(rule[1], list) else [rule[1]]
                if pos <= len(seq):
                    actual_aa = seq[pos-1]
                    if actual_aa in targets:
                        mut_str = f"Del {pos}" if actual_aa == '-' else f"{pos}{actual_aa}"
                        fmt_str = format_mut_display(gene, mut_str)
                        if (gene, str(pos), actual_aa) in KEY_TARGETS:
                            key_muts.append(fmt_str)
                        else:
                            other_muts.append(fmt_str)

    # Biological Segment Sorting Logic
    def sort_mutations(mut_list):
        seg_order = {'PB2': 0, 'PB1': 1, 'PA': 2, 'HA': 3, 'NP': 4, 'NA': 5, 'M': 6, 'NS': 7}
        def sort_key(m):
            seg = m.split('_')[0]
            order = seg_order.get(seg, 99)
            return (order, m)
        return sorted(list(set(mut_list)), key=sort_key)

    return pd.DataFrame([features])[sel_cols], sort_mutations(key_muts), sort_mutations(other_muts)

# ================= 6. User Interface =================
st.title("H5 Sentinel: Mammalian Adaptation Risk Assessment System")
st.markdown("**A Machine Learning-Based Zoonotic Risk Surveillance Framework for Avian Influenza**")
st.markdown("---")

with st.sidebar:
    st.header("Analysis Parameters")
    
    seq_type = st.radio("Sequence Type:", ["Amino Acid (Protein)", "Nucleotide (DNA/RNA)"])
    is_nucleotide = (seq_type == "Nucleotide (DNA/RNA)")
    
    target_segments = st.selectbox(
        "Target Segment Scope:",
        ["Full-length (8 Segments)", "HA Only", "PB2 Only"]
    )
    
    st.markdown("---")
    st.info(f"**Risk Stratification Thresholds:**\n\n"
            f"- **Natural Reservoir:** MARS < {T_LOW}\n"
            f"- **Pre-adaptation:** {T_LOW} - {T_HIGH}\n"
            f"- **Imminent Spillover:** MARS >= {T_HIGH}")
    st.markdown("---")
    st.markdown("**Batch Analysis Mode:**\n\n"
                "The framework permits direct ingestion of multi-FASTA files for high-throughput surveillance. "
                "An automated risk assessment profile and quantitative CSV export will be dynamically compiled.\n\n"
                "**Formatting Standard:**\n"
                "Sequence headers must conform to the `>Segment|StrainName` nomenclature to ensure precise segment allocation.\n\n"
                "*Example Structure:*\n"
                "`>HA|A/chicken/Florida/22_033392_001/2022`\n"
                "`>PB2|A/chicken/Florida/22_033392_001/2022`\n"
                "`>NP|A/chicken/Florida/22_033392_001/2022`")

model, sel_cols = load_models(target_segments)

uploaded_file = st.file_uploader("Upload FASTA Sequence (.fasta, .fas, .txt)", type=['fasta', 'fas', 'txt'])

if st.button("Execute Batch Prediction", type="primary"):
    if uploaded_file is None:
        st.warning("Please specify a valid FASTA format dataset to proceed.")
    elif model is None:
        st.error("Model initialization failure. Please verify internal dependency paths for the selected segment architecture.")
    else:
        fasta_input = uploaded_file.getvalue().decode("utf-8")
        
        with st.spinner("Processing genomic assemblies and mapping physicochemical feature matrices..."):
            strains_dict = process_fasta_multi(fasta_input, is_nucleotide)
            
            valid_strains = {}
            for strain_id, seq_dict in strains_dict.items():
                if target_segments == "HA Only" and 'HA' not in seq_dict:
                    continue
                if target_segments == "PB2 Only" and 'PB2' not in seq_dict:
                    continue
                if target_segments == "Full-length (8 Segments)" and len(seq_dict) == 0:
                    continue
                valid_strains[strain_id] = seq_dict

            if not valid_strains:
                st.error("Data Compilation Error: No valid genomic sequences matching the specified segment criteria were identified.")
            else:
                total_strains = len(valid_strains)
                st.success(f"System successfully resolved {total_strains} discrete viral isolates. Model inference initiated.")
                
                results_summary = []
                detailed_results = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (strain_id, seq_dict) in enumerate(valid_strains.items()):
                    X_pred, key_muts, other_muts = extract_features(seq_dict, sel_cols)
                    mars_score = model.predict_proba(X_pred)[0, 1]
                    
                    if mars_score < T_LOW:
                        tier, color_hex = "Low Risk", "#28a745"
                    elif mars_score < T_HIGH:
                        tier, color_hex = "Moderate Risk", "#ffc107"
                    else:
                        tier, color_hex = "High Risk", "#dc3545"
                        
                    genes_found = ", ".join(seq_dict.keys())
                    
                    results_summary.append({
                        "Strain ID": strain_id,
                        "Detected Segments": genes_found,
                        "MARS Score": round(mars_score, 4),
                        "Risk Tier": tier
                    })
                    
                    detailed_results[strain_id] = {
                        "score": mars_score,
                        "tier": tier,
                        "color": color_hex,
                        "key_muts": key_muts,
                        "other_muts": other_muts,
                        "genes": genes_found
                    }
                    
                    progress_percent = int((i + 1) / total_strains * 100)
                    progress_bar.progress(progress_percent)
                    status_text.text(f"Evaluated: Isolate {i + 1} of {total_strains}")
                
                status_text.text("Quantitative inference concluded. Rendering predictive reports.")
                
                df_summary = pd.DataFrame(results_summary)
                df_summary = df_summary.sort_values(by="MARS Score", ascending=False).reset_index(drop=True)
                
                st.markdown("---")
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.subheader(f"Batch Quantitative Assessment (n = {total_strains})")
                with col_b:
                    csv_data = df_summary.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="Export Evaluation Matrix (CSV)",
                        data=csv_data,
                        file_name="MARS_Risk_Evaluation_Matrix.csv",
                        mime="text/csv",
                        type="primary"
                    )
                
                st.dataframe(df_summary, use_container_width=True)
                
                st.markdown("---")
                render_limit = min(50, total_strains)
                st.subheader(f"Molecular Diagnostic Profiling (Top {render_limit} Iterations)")
                if total_strains > 50:
                    st.info("System Note: Diagnostic sub-dashboards are constrained to the top 50 prioritized isolates to maintain rendering integrity. Comprehensive metrics are securely accessible via the CSV export.")
                
                for i in range(render_limit):
                    strain_id = df_summary.iloc[i]["Strain ID"]
                    data = detailed_results[strain_id]
                    mars = data["score"]
                    
                    is_expanded = True if mars >= T_HIGH else False
                    
                    # Clean title without markdown asterisks to prevent rendering bugs in standard Streamlit expanders
                    with st.expander(f"Rank #{i+1}: {strain_id} [ MARS: {mars:.3f} | {data['tier']} ]", expanded=is_expanded):
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number", value = mars,
                                number = {'font': {'size': 42}}, # Enforced max size to prevent horizontal overflow
                                title = {'text': "<b>MARS Score</b>", 'font': {'size': 20}},
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                gauge = {
                                    'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                    'bar': {'color': "rgba(0,0,0,0)"},
                                    'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                                    'steps': [
                                        {'range': [0, T_LOW], 'color': '#d4edda'},
                                        {'range': [T_LOW, T_HIGH], 'color': '#fff3cd'},
                                        {'range': [T_HIGH, 1.0], 'color': '#f8d7da'}
                                    ],
                                    'threshold': {'line': {'color': "black", 'width': 5}, 'thickness': 0.8, 'value': mars}
                                }
                            ))
                            # Expanded top/bottom margins and total height to securely encompass the title and gauge boundaries
                            fig.update_layout(height=320, margin=dict(l=30, r=30, t=65, b=30), font=dict(family="Arial"))
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with col2:
                            st.markdown(f"#### Pathogenic Assessment: <span style='color:{data['color']}; font-weight:bold'>{data['tier']}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Assembled Segments:** `{data['genes']}`")
                            
                            st.markdown("**Key Adaptation Signatures Detected:**")
                            if data["key_muts"]:
                                st.markdown(", ".join([f"**{m}**" for m in data["key_muts"]]))
                            else:
                                st.markdown("_No primary adaptation markers isolated._")
                                
                            st.markdown("**Other Literature-Reported Adaptation Sites Detected:**")
                            if data["other_muts"]:
                                st.markdown(", ".join([f"{m}" for m in data["other_muts"]]))
                            else:
                                st.markdown("_No subsidiary adaptation markers isolated._")