import streamlit as st
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd
import py3Dmol
from stmol import showmol
import random

##############################################
# Helper functions for atom types and chain colors
##############################################

# Dictionary: three-letter to one-letter amino acid codes.
three_to_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLU": "E", "GLN": "Q", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

# Backbone atoms common to all residues.
backbone_atoms = {"N", "CA", "C", "O", "OXT"}

# Side-chain atoms by residue.
side_chain_atoms = {
    "ALA": {"CB"},
    "ARG": {"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"CB", "CG", "OD1", "ND2"},
    "ASP": {"CB", "CG", "OD1", "OD2"},
    "CYS": {"CB", "SG"},
    "GLU": {"CB", "CG", "CD", "OE1", "OE2"},
    "GLN": {"CB", "CG", "CD", "OE1", "NE2"},
    "HIS": {"CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"},
    "LEU": {"CB", "CG", "CD1", "CD2"},
    "LYS": {"CB", "CG", "CD", "CE", "NZ"},
    "MET": {"CB", "CG", "SD", "CE"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"CB", "CG", "CD"},
    "SER": {"CB", "OG"},
    "THR": {"CB", "OG1", "CG2"},
    "TRP": {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"CB", "CG1", "CG2"}
}

def determine_atom_category(res, atname):
    """
    Determine whether the atom (atname) in residue (res) is Backbone or Side chain.
    """
    atom = atname.strip().upper()
    if atom in backbone_atoms:
        return "Backbone"
    if res.upper() in side_chain_atoms and atom in side_chain_atoms[res.upper()]:
        return "Side chain"
    return "Side chain"

def determine_interaction_subtype(cat1, cat2):
    """
    Return a string indicating the interaction subtype.
    """
    if cat1 == cat2:
        return f"{cat1} - {cat1}"
    else:
        return "Side chain - Backbone"

def generate_chain_colors(chains):
    """
    Generate a unique random color for each chain.
    Returns a dictionary mapping chain IDs to hex color strings.
    """
    colors = ["#{:06x}".format(random.randint(0, 0xFFFFFF)) for _ in range(len(chains))]
    return dict(zip(chains, colors))

##############################################
# XML Parsing Functions
##############################################

# ----- Parsing for Whole Interface Tab (Tab 1) -----
def parse_xml_to_dict_tab1(file_obj):
    """
    Parse the PISA XML file using a simple recursive parser.
    Uses the order as in the XML (chain_pair is not sorted).
    """
    file_obj.seek(0)  # Reset file pointer
    def recursive_parse(element):
        parsed_data = {}
        for child in element:
            if len(child) > 0:
                parsed_data[child.tag] = recursive_parse(child)
            else:
                parsed_data[child.tag] = child.text.strip() if child.text else None
        return parsed_data

    tree = ET.parse(file_obj)
    root = tree.getroot()
    interfaces = defaultdict(lambda: defaultdict(lambda: {"h-bonds": [], "salt-bridges": []}))
    for interface in root.findall('interface'):
        interface_id = interface.find('id').text.strip()
        # Process h-bonds.
        h_bonds = interface.find('h-bonds')
        if h_bonds is not None:
            for bond in h_bonds.findall("bond"):
                bond_data = recursive_parse(bond)
                chain_1 = bond_data.pop('chain-1')
                chain_2 = bond_data.pop('chain-2')
                chain_pair = (chain_1, chain_2)
                interfaces[interface_id][chain_pair]["h-bonds"].append(bond_data)
        # Process salt-bridges.
        salt_bridges = interface.find('salt-bridges')
        if salt_bridges is not None:
            for bond in salt_bridges.findall("bond"):
                bond_data = recursive_parse(bond)
                chain_1 = bond_data.pop('chain-1')
                chain_2 = bond_data.pop('chain-2')
                chain_pair = (chain_1, chain_2)
                interfaces[interface_id][chain_pair]["salt-bridges"].append(bond_data)
    return {k: dict(v) for k, v in interfaces.items()}

# ----- Detailed Parsing for Interface Details Tab (Tab 2) -----
def parse_xml_to_dict_detailed(file_obj):
    """
    Parse the PISA XML file and convert it into a nested dictionary.
    For each bond, preserve the original chain values in "orig_chain-1"/"orig_chain-2"
    and group bonds using a sorted tuple.
    """
    file_obj.seek(0)  # Reset file pointer
    def recursive_parse(element):
        parsed_data = {}
        for child in element:
            if len(child) > 0:
                parsed_data[child.tag] = recursive_parse(child)
            else:
                parsed_data[child.tag] = child.text.strip() if child.text else None
        return parsed_data

    tree = ET.parse(file_obj)
    root = tree.getroot()
    interfaces = defaultdict(lambda: defaultdict(lambda: {"h-bonds": [], "salt-bridges": []}))
    for interface in root.findall('interface'):
        interface_id = interface.find('id').text.strip()
        # Process hydrogen bonds.
        h_bonds = interface.find('h-bonds')
        if h_bonds is not None:
            for bond in h_bonds.findall("bond"):
                bond_data = recursive_parse(bond)
                orig_chain1 = bond_data.get('chain-1')
                orig_chain2 = bond_data.get('chain-2')
                bond_data["orig_chain-1"] = orig_chain1
                bond_data["orig_chain-2"] = orig_chain2
                chain_1 = bond_data.pop('chain-1')
                chain_2 = bond_data.pop('chain-2')
                chain_pair = tuple(sorted([chain_1, chain_2]))
                interfaces[interface_id][chain_pair]["h-bonds"].append(bond_data)
        # Process salt-bridges.
        salt_bridges = interface.find('salt-bridges')
        if salt_bridges is not None:
            for bond in salt_bridges.findall("bond"):
                bond_data = recursive_parse(bond)
                orig_chain1 = bond_data.get('chain-1')
                orig_chain2 = bond_data.get('chain-2')
                bond_data["orig_chain-1"] = orig_chain1
                bond_data["orig_chain-2"] = orig_chain2
                chain_1 = bond_data.pop('chain-1')
                chain_2 = bond_data.pop('chain-2')
                chain_pair = tuple(sorted([chain_1, chain_2]))
                interfaces[interface_id][chain_pair]["salt-bridges"].append(bond_data)
    return {k: dict(v) for k, v in interfaces.items()}

##############################################
# Functions for Whole Interface Tab (Tab 1)
##############################################

def create_interactions_dataframe(interactions, chain_pair):
    """
    Create a DataFrame of interactions.
    Each row shows:
      - Monomer 1: chain_1, res-1, seqnum-1, atname-1
      - Distance, Monomer 2: chain_2, res-2, seqnum-2, atname-2, and Interaction Type.
    """
    data = []
    chain_1, chain_2 = chain_pair
    for bond_type, bonds in interactions.items():
        interaction_type = "Hydrogen Bond" if bond_type == 'h-bonds' else "Salt Bridge"
        for bond in bonds:
            monomer_1 = f"{chain_1} {bond.get('res-1', 'N/A')} {bond.get('seqnum-1', 'N/A')} [{bond.get('atname-1', 'N/A')}]"
            monomer_2 = f"{chain_2} {bond.get('res-2', 'N/A')} {bond.get('seqnum-2', 'N/A')} [{bond.get('atname-2', 'N/A')}]"
            data.append([monomer_1, bond.get('dist', 'N/A'), monomer_2, interaction_type])
    return pd.DataFrame(data, columns=["Monomer 1", "Length", "Monomer 2", "Interaction Type"])

def generate_chimerax_command_tab1(interface_data):
    """
    Generate a list of ChimeraX commands for the given interface data.
    (For each chain pair in the interface.)
    """
    chimerax_commands = []
    for chain_pair, interactions in interface_data.items():
        chain_1, chain_2 = chain_pair
        residues_chain_1 = set()
        residues_chain_2 = set()
        for bond in interactions.get('h-bonds', []):
            residues_chain_1.add(bond['seqnum-1'])
            residues_chain_2.add(bond['seqnum-2'])
        for bond in interactions.get('salt-bridges', []):
            residues_chain_1.add(bond['seqnum-1'])
            residues_chain_2.add(bond['seqnum-2'])
        chain_1_command = f"sel #1/{chain_1}:{':'.join(sorted(residues_chain_1))}"
        chain_2_command = f"sel #1/{chain_2}:{':'.join(sorted(residues_chain_2))}"
        hbonds_command = (
            f"cartoon suppressBackboneDisplay false\n"
            f"hbonds (#1/{chain_1} & protein) restrict (#1/{chain_2} & protein) "
            f"reveal true relax true distSlop 0.8 angleSlop 25"
        )
        chimerax_commands.extend([chain_1_command, chain_2_command, hbonds_command])
    return chimerax_commands

def visualize_pdb(pdb_data, chain_pair, residues, chain_colors):
    """
    Visualize the PDB structure with colored chains and highlighted residues.
    """
    viewer = py3Dmol.view()
    viewer.addModel(pdb_data, "pdb")
    for chain, color in chain_colors.items():
        viewer.setStyle({"chain": chain}, {"cartoon": {"color": color}})
    for chain, res_list in residues.items():
        for res_num in res_list:
            sel = {"chain": chain, "resi": res_num, "byres": True}
            viewer.setStyle(sel, {"stick": {"color": chain_colors[chain]}})
    viewer.zoomTo({"chain": list(chain_colors.keys())})
    showmol(viewer, height=800, width=800)

##############################################
# Functions for Interface Details Tab (Tab 2)
##############################################

def find_residues_and_generate_chimerax_command(interfaces, group1, group2):
    """
    Generate a ChimeraX selection command using detailed interface data.
    """
    interacting_residues = set()
    for interface_id, interface_data in interfaces.items():
        for chain_pair, interactions in interface_data.items():
            for bond in interactions.get('h-bonds', []):
                orig_chain1 = bond.get("orig_chain-1")
                orig_chain2 = bond.get("orig_chain-2")
                seqnum1 = bond.get('seqnum-1')
                seqnum2 = bond.get('seqnum-2')
                if seqnum1 and seqnum2:
                    if orig_chain1 in group1 and orig_chain2 in group2:
                        interacting_residues.add(f"{orig_chain1}:{seqnum1}")
                        interacting_residues.add(f"{orig_chain2}:{seqnum2}")
                    elif orig_chain1 in group2 and orig_chain2 in group1:
                        interacting_residues.add(f"{orig_chain2}:{seqnum2}")
                        interacting_residues.add(f"{orig_chain1}:{seqnum1}")
            for bond in interactions.get('salt-bridges', []):
                orig_chain1 = bond.get("orig_chain-1")
                orig_chain2 = bond.get("orig_chain-2")
                seqnum1 = bond.get('seqnum-1')
                seqnum2 = bond.get('seqnum-2')
                if seqnum1 and seqnum2:
                    if orig_chain1 in group1 and orig_chain2 in group2:
                        interacting_residues.add(f"{orig_chain1}:{seqnum1}")
                        interacting_residues.add(f"{orig_chain2}:{seqnum2}")
                    elif orig_chain1 in group2 and orig_chain2 in group1:
                        interacting_residues.add(f"{orig_chain2}:{seqnum2}")
                        interacting_residues.add(f"{orig_chain1}:{seqnum1}")
    if not interacting_residues:
        return "# No interacting residues found."
    residues_by_chain = defaultdict(set)
    for residue in interacting_residues:
        chain, seqnum = residue.split(":")
        residues_by_chain[chain].add(seqnum)
    select_parts = []
    for chain in sorted(residues_by_chain.keys()):
        residues = residues_by_chain[chain]
        try:
            sorted_residues = sorted(residues, key=lambda x: int(x))
        except ValueError:
            sorted_residues = sorted(residues)
        residue_list = ",".join(sorted_residues)
        select_parts.append(f"/{chain}:{residue_list}")
    return "select " + "".join(select_parts)

def generate_interaction_table(interfaces, group1, group2):
    """
    Generate a detailed interaction table.
    Returns a list of tuples: (Selection 1, Selection 2, Interaction Type, Atom Interaction)
    """
    table = defaultdict(set)
    for interface_id, interface_data in interfaces.items():
        for chain_pair, interactions in interface_data.items():
            for bond in interactions.get("h-bonds", []):
                orig_chain1 = bond.get("orig_chain-1")
                orig_chain2 = bond.get("orig_chain-2")
                res1 = bond.get("res-1", "").strip() if bond.get("res-1") else ""
                res2 = bond.get("res-2", "").strip() if bond.get("res-2") else ""
                seqnum1 = bond.get("seqnum-1", "").strip() if bond.get("seqnum-1") else ""
                seqnum2 = bond.get("seqnum-2", "").strip() if bond.get("seqnum-2") else ""
                atname1 = bond.get("atname-1", "").strip() if bond.get("atname-1") else ""
                atname2 = bond.get("atname-2", "").strip() if bond.get("atname-2") else ""
                aa1 = three_to_one.get(res1.upper(), res1.upper()[0]) if res1 else ""
                aa2 = three_to_one.get(res2.upper(), res2.upper()[0]) if res2 else ""
                if orig_chain1 in group1 and orig_chain2 in group2:
                    sel1 = f"{orig_chain1} {aa1}{seqnum1}"
                    sel2 = f"{orig_chain2} {aa2}{seqnum2}"
                    r1, a1 = res1, atname1
                    r2, a2 = res2, atname2
                elif orig_chain1 in group2 and orig_chain2 in group1:
                    sel1 = f"{orig_chain2} {aa2}{seqnum2}"
                    sel2 = f"{orig_chain1} {aa1}{seqnum1}"
                    r1, a1 = res2, atname2
                    r2, a2 = res1, atname1
                else:
                    continue
                cat1 = determine_atom_category(r1, a1)
                cat2 = determine_atom_category(r2, a2)
                subtype = determine_interaction_subtype(cat1, cat2)
                table[(sel1, "Hydrogen bond", subtype)].add(sel2)
            for bond in interactions.get("salt-bridges", []):
                orig_chain1 = bond.get("orig_chain-1")
                orig_chain2 = bond.get("orig_chain-2")
                res1 = bond.get("res-1", "").strip() if bond.get("res-1") else ""
                res2 = bond.get("res-2", "").strip() if bond.get("res-2") else ""
                seqnum1 = bond.get("seqnum-1", "").strip() if bond.get("seqnum-1") else ""
                seqnum2 = bond.get("seqnum-2", "").strip() if bond.get("seqnum-2") else ""
                atname1 = bond.get("atname-1", "").strip() if bond.get("atname-1") else ""
                atname2 = bond.get("atname-2", "").strip() if bond.get("atname-2") else ""
                aa1 = three_to_one.get(res1.upper(), res1.upper()[0]) if res1 else ""
                aa2 = three_to_one.get(res2.upper(), res2.upper()[0]) if res2 else ""
                if orig_chain1 in group1 and orig_chain2 in group2:
                    sel1 = f"{orig_chain1} {aa1}{seqnum1}"
                    sel2 = f"{orig_chain2} {aa2}{seqnum2}"
                    r1, a1 = res1, atname1
                    r2, a2 = res2, atname2
                elif orig_chain1 in group2 and orig_chain2 in group1:
                    sel1 = f"{orig_chain2} {aa2}{seqnum2}"
                    sel2 = f"{orig_chain1} {aa1}{seqnum1}"
                    r1, a1 = res2, atname2
                    r2, a2 = res1, atname1
                else:
                    continue
                cat1 = determine_atom_category(r1, a1)
                cat2 = determine_atom_category(r2, a2)
                subtype = determine_interaction_subtype(cat1, cat2)
                table[(sel1, "Salt bridge", subtype)].add(sel2)
    rows = []
    for (sel1, itype, subtype), sel2_set in table.items():
        sel2_str = ", ".join(sorted(list(sel2_set)))
        rows.append((sel1, sel2_str, itype, subtype))
    def extract_res_num(s):
        try:
            parts = s.split()
            if len(parts) < 2:
                return 0
            num_str = "".join(filter(str.isdigit, parts[1]))
            return int(num_str)
        except:
            return 0
    rows = sorted(rows, key=lambda x: extract_res_num(x[0]))
    return rows

def get_unique_chains(interfaces):
    """
    Extract unique chain IDs from detailed interface data.
    """
    unique = set()
    for interface_id, interface_data in interfaces.items():
        for chain_pair, interactions in interface_data.items():
            for bond in interactions.get('h-bonds', []):
                if bond.get("orig_chain-1"):
                    unique.add(bond.get("orig_chain-1"))
                if bond.get("orig_chain-2"):
                    unique.add(bond.get("orig_chain-2"))
            for bond in interactions.get('salt-bridges', []):
                if bond.get("orig_chain-1"):
                    unique.add(bond.get("orig_chain-1"))
                if bond.get("orig_chain-2"):
                    unique.add(bond.get("orig_chain-2"))
    return sorted(unique)

def modify_chain_label(cell, hide, mapping):
    """
    Modify a cell string (e.g. "G G102") by either hiding the chain letter
    or replacing it using a mapping.
    """
    parts = cell.split()
    if len(parts) < 2:
        return cell
    chain = parts[0]
    rest = " ".join(parts[1:])
    if hide:
        return rest
    elif mapping and chain in mapping:
        return f"{mapping[chain]} {rest}"
    else:
        return cell

##############################################
# Streamlit App Layout
##############################################

st.set_page_config(layout="wide")
st.title("PISA XML Interface Explorer with ChimeraX & 3D View")

# Move file uploaders to the sidebar for an uncluttered main view.
st.sidebar.header("Upload Files")
uploaded_pisa = st.sidebar.file_uploader("Upload PISA XML File", type=["xml", "intf"])
uploaded_pdb = st.sidebar.file_uploader("Upload PDB File (Optional)", type=["pdb"])

if uploaded_pisa:
    # Create two tabs.
    tab1, tab2 = st.tabs(["Whole Interface", "Interface Details"])
    
    ##########################
    # Tab 1: Whole Interface
    ##########################
    with tab1:
        st.header("Whole Interface View")
        interfaces_tab1 = parse_xml_to_dict_tab1(uploaded_pisa)
        interface_ids = list(interfaces_tab1.keys())
        if not interface_ids:
            st.info("No interfaces found in the uploaded XML file.")
        else:
            # Use two columns: left for selections, right for structure visualization.
            col1, col2 = st.columns([1, 2])
            with col1:
                selected_interface_id = st.selectbox("Interface ID", interface_ids)
                interface_data_tab1 = interfaces_tab1[selected_interface_id]
                chain_pair_options = list(interface_data_tab1.keys())
                if chain_pair_options:
                    selected_chain_pair = st.selectbox("Chain Pair", chain_pair_options)
                else:
                    st.info("No chain pairs found in the selected interface.")
            if chain_pair_options:
                chain1, chain2 = selected_chain_pair
                with col2:
                    st.markdown("### Chain Color Selection")
                    color_chain1 = st.color_picker(f"Color for Chain {chain1}", "#659885")
                    color_chain2 = st.color_picker(f"Color for Chain {chain2}", "#986578")
                    chain_colors = {chain1: color_chain1, chain2: color_chain2}
                    if uploaded_pdb:
                        st.subheader("3D Structure Visualization")
                        # Reset file pointer for pdb file if needed.
                        uploaded_pdb.seek(0)
                        pdb_text = uploaded_pdb.read().decode("utf-8")
                        residues_to_highlight = defaultdict(list)
                        for bond_type, bonds in interface_data_tab1[selected_chain_pair].items():
                            for bond in bonds:
                                try:
                                    residues_to_highlight[chain1].append(int(bond['seqnum-1']))
                                except:
                                    pass
                                try:
                                    residues_to_highlight[chain2].append(int(bond['seqnum-2']))
                                except:
                                    pass
                        visualize_pdb(pdb_text, selected_chain_pair, residues_to_highlight, chain_colors)
                st.markdown("---")
                st.subheader("Interactions Overview")
                interactions_df = create_interactions_dataframe(interface_data_tab1[selected_chain_pair],
                                                                  selected_chain_pair)
                st.dataframe(interactions_df)
                st.subheader("ChimeraX Commands")
                commands = generate_chimerax_command_tab1(interface_data_tab1)
                for cmd in commands:
                    st.code(cmd, language="bash")
    
    ##########################
    # Tab 2: Interface Details
    ##########################
    with tab2:
        st.header("Detailed Interface View")
        interfaces_detailed = parse_xml_to_dict_detailed(uploaded_pisa)
        unique_chains = get_unique_chains(interfaces_detailed)
        st.markdown("**Detected Chains:** " + ", ".join(unique_chains))
        # Use two columns for group selections.
        col_group1, col_group2 = st.columns(2)
        with col_group1:
            group1 = st.multiselect("Select chains for Group 1 (e.g. antigen)", unique_chains)
        with col_group2:
            group2 = st.multiselect("Select chains for Group 2 (e.g. antibody)", unique_chains)
        st.markdown("### Chain Label Options")
        hide_chain = st.checkbox("Hide chain names in table", value=False)
        custom_mapping_input = st.text_input("Custom chain label mapping (e.g. A:Antigen, B:Antibody)", value="")
        custom_mapping = {}
        if custom_mapping_input and not hide_chain:
            for pair in custom_mapping_input.split(","):
                parts = pair.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    custom_mapping[key] = val
        
        if group1 and group2:
            chimera_command_detailed = find_residues_and_generate_chimerax_command(interfaces_detailed, group1, group2)
            st.subheader("ChimeraX Selection Command")
            st.code(chimera_command_detailed, language="bash")
            
            table_rows = generate_interaction_table(interfaces_detailed, group1, group2)
            if table_rows:
                modified_rows = []
                for row in table_rows:
                    sel1_mod = modify_chain_label(row[0], hide_chain, custom_mapping)
                    sel2_mod = modify_chain_label(row[1], hide_chain, custom_mapping)
                    modified_rows.append((sel1_mod, sel2_mod, row[2], row[3]))
                df_details = pd.DataFrame(modified_rows, columns=["Selection 1", "Selection 2", "Interaction Type", "Atom Interaction"])
                st.subheader("Interaction Details Table")
                st.dataframe(df_details)
            else:
                st.warning("No interactions found for the selected groups.")
        else:
            st.info("Please select at least one chain for each group.")
else:
    st.info("Awaiting PISA XML file upload.")
