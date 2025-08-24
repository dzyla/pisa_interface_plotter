import streamlit as st
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd
import tempfile
import os
from streamlit_molstar import st_molstar

##############################################
# Helper Functions
##############################################

# Dictionary: three-letter to one-letter amino acid codes.
three_to_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V"
}

# Backbone atoms common to all residues.
backbone_atoms = {"N", "CA", "C", "O", "OXT"}

# Side-chain atoms by residue.
side_chain_atoms = {
    "ALA": {"CB"}, "ARG": {"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"CB", "CG", "OD1", "ND2"}, "ASP": {"CB", "CG", "OD1", "OD2"},
    "CYS": {"CB", "SG"}, "GLU": {"CB", "CG", "CD", "OE1", "OE2"},
    "GLN": {"CB", "CG", "CD", "OE1", "NE2"}, "HIS": {"CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"}, "LEU": {"CB", "CG", "CD1", "CD2"},
    "LYS": {"CB", "CG", "CD", "CE", "NZ"}, "MET": {"CB", "CG", "SD", "CE"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"}, "PRO": {"CB", "CG", "CD"},
    "SER": {"CB", "OG"}, "THR": {"CB", "OG1", "CG2"},
    "TRP": {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"}, "VAL": {"CB", "CG1", "CG2"}
}


def determine_atom_category(res, atname):
    atom = atname.strip().upper()
    if atom in backbone_atoms: return "Backbone"
    if res.upper() in side_chain_atoms and atom in side_chain_atoms[res.upper()]: return "Side chain"
    return "Side chain"

def determine_interaction_subtype(cat1, cat2):
    return f"{cat1} - {cat2}"

##############################################
# Main Data Parsing and Processing Functions
##############################################

def parse_pisa_xml(file_obj):
    file_obj.seek(0)
    tree = ET.parse(file_obj)
    root = tree.getroot()

    pisa_data = {'stats': [], 'interfaces': {}, 'all_chains': set()}

    def recursive_parse(element):
        return {child.tag: recursive_parse(child) if len(child) > 0 else (child.text.strip() if child.text else None) for child in element}

    for interface_node in root.findall('interface'):
        interface_id = int(interface_node.find('id').text)
        
        stat = {'id': interface_id}
        for tag in ['int_area', 'int_solv_en', 'stab_en', 'pvalue']:
            node = interface_node.find(tag)
            if node is not None and node.text:
                try:
                    stat[tag] = float(node.text)
                except (ValueError, TypeError):
                    stat[tag] = node.text

        molecules_in_interface = []
        for mol_node in interface_node.findall('molecule'):
            bsa_node = mol_node.find('bsa')
            bsa_value = float(bsa_node.text) if bsa_node is not None and bsa_node.text is not None else 0.0
            
            mol_info = {
                'chain_id': mol_node.find('chain_id').text,
                'bsa': bsa_value
            }
            molecules_in_interface.append(f"{mol_info['chain_id']}")
        stat['molecules'] = '; '.join(molecules_in_interface)
        pisa_data['stats'].append(stat)

        interface_details = {'bonds': [], 'residues': defaultdict(list)}
        
        for bond_type in ['h-bonds', 'salt-bridges']:
            bond_element = interface_node.find(bond_type)
            if bond_element is not None:
                for bond in bond_element.findall("bond"):
                    bond_data = recursive_parse(bond)
                    c1, c2 = bond_data.get('chain-1'), bond_data.get('chain-2')
                    if not c1 or not c2: continue
                    
                    bond_data['type'] = "Hydrogen bond" if bond_type == "h-bonds" else "Salt bridge"
                    interface_details['bonds'].append(bond_data)

                    pisa_data['all_chains'].update([c1, c2])
                    try:
                        interface_details['residues'][c1].append(int(bond_data['seqnum-1']))
                        interface_details['residues'][c2].append(int(bond_data['seqnum-2']))
                    except (ValueError, TypeError):
                        pass
        pisa_data['interfaces'][interface_id] = interface_details

    pisa_data['all_chains'] = sorted(list(pisa_data['all_chains']))
    return pisa_data

def generate_interaction_table(bonds, group1, group2, chain_id_to_label, show_chain_id=True, group_by='group1', group_identical=False):

    def get_residue_identifier(res_str):
        parts = res_str.split()
        return " ".join(parts[1:]) if len(parts) > 1 and len(parts[0]) == 1 and parts[0].isupper() else res_str

    raw_bonds = set()
    for bond in bonds:
        c1, c2 = bond.get("chain-1"), bond.get("chain-2")
        r1, r2 = bond.get("res-1", "").strip(), bond.get("res-2", "").strip()
        s1, s2 = bond.get("seqnum-1", "").strip(), bond.get("seqnum-2", "").strip()
        a1, a2 = bond.get("atname-1", "").strip(), bond.get("atname-2", "").strip()

        if (c1 in group1 and c2 in group2): p1_chain, p2_chain, p1_res, p2_res, p1_seq, p2_seq, p1_at, p2_at = c1, c2, r1, r2, s1, s2, a1, a2
        elif (c1 in group2 and c2 in group1): p1_chain, p2_chain, p1_res, p2_res, p1_seq, p2_seq, p1_at, p2_at = c2, c1, r2, r1, s2, s1, a2, a1
        else: continue

        p1_aa, p2_aa = three_to_one.get(p1_res.upper(), "?"), three_to_one.get(p2_res.upper(), "?")
        p1_type, p2_type = chain_id_to_label.get(p1_chain, p1_chain), chain_id_to_label.get(p2_chain, p2_chain)
        res1_str = f"{p1_chain} {p1_aa}{p1_seq}" if show_chain_id else f"{p1_aa}{p1_seq}"
        res2_str = f"{p2_chain} {p2_aa}{p2_seq}" if show_chain_id else f"{p2_aa}{p2_seq}"
        subtype = determine_interaction_subtype(determine_atom_category(p1_res, p1_at), determine_atom_category(p2_res, p2_at))
        raw_bonds.add((p1_type, res1_str, p2_type, res2_str, bond['type'], subtype))

    if not raw_bonds: return []

    if group_by == 'none':
        rows = list(raw_bonds)
    else:
        table = defaultdict(set)
        if group_by == 'group1':
            for p1, r1, p2, r2, itype, subtype in raw_bonds: table[(p1, r1, itype, subtype)].add((p2, r2))
        elif group_by == 'group2':
            for p1, r1, p2, r2, itype, subtype in raw_bonds: table[(p2, r2, itype, subtype)].add((p1, r1))

        rows = []
        for key, partner_set in table.items():
            partner_by_type = defaultdict(list)
            for p_type, res_str in partner_set: partner_by_type[p_type].append(res_str)
            partner_type_str = ", ".join(sorted(partner_by_type.keys()))
            partner_res_list_full = [res for p_type in sorted(partner_by_type.keys()) for res in sorted(partner_by_type[p_type])]
            
            if group_identical:
                unique_partners = {get_residue_identifier(res_str): res_str for res_str in reversed(partner_res_list_full)}
                final_partner_list = list(unique_partners.values())
            else:
                final_partner_list = partner_res_list_full
            partner_res_str = ", ".join(final_partner_list)

            if group_by == 'group1':
                p1_type, res1_str, itype, subtype = key
                rows.append((p1_type, res1_str, partner_type_str, partner_res_str, itype, subtype))
            elif group_by == 'group2':
                p2_type, res2_str, itype, subtype = key
                rows.append((partner_type_str, partner_res_str, p2_type, res2_str, itype, subtype))

    if group_identical:
        merged_rows_dict = {}
        for row in rows:
            p1, r1, p2, r2, itype, sub = row
            key_map = {
                'group1': (p1, get_residue_identifier(r1), p2, r2, itype, sub),
                'group2': (p1, r1, p2, get_residue_identifier(r2), itype, sub),
                'none': (p1, get_residue_identifier(r1), p2, get_residue_identifier(r2), itype, sub)
            }
            key = key_map[group_by]
            if key not in merged_rows_dict: merged_rows_dict[key] = row
        rows = list(merged_rows_dict.values())
    
    return rows

def find_residues_and_generate_chimerax_command(bonds, group1, group2):
    interacting_residues = set()
    for bond in bonds:
        c1, c2 = bond.get("chain-1"), bond.get("chain-2")
        s1, s2 = bond.get('seqnum-1'), bond.get('seqnum-2')
        if s1 and s2:
            if (c1 in group1 and c2 in group2) or (c1 in group2 and c2 in group1):
                interacting_residues.add(f"{c1}:{s1}")
                interacting_residues.add(f"{c2}:{s2}")
    if not interacting_residues: return "# No interacting residues found."

    residues_by_chain = defaultdict(set)
    for res in interacting_residues:
        chain, seqnum = res.split(":")
        residues_by_chain[chain].add(seqnum)

    select_parts = []
    for chain, residues in sorted(residues_by_chain.items()):
        try: sorted_residues = sorted(list(residues), key=int)
        except ValueError: sorted_residues = sorted(list(residues))
        select_parts.append(f"/{chain}:{','.join(sorted_residues)}")
    return "select " + "".join(select_parts)

##############################################
# Streamlit App Layout
##############################################

st.set_page_config(layout="wide")
st.title("PISA Interface Explorer 🔬")

st.sidebar.header("📁 Upload Files")
uploaded_pisa = st.sidebar.file_uploader("Upload PISA XML File", type=["xml", "intf"])
uploaded_pdb = st.sidebar.file_uploader("Upload PDB File (for 3D view)", type=["pdb"])

if uploaded_pisa:
    pisa_data = parse_pisa_xml(uploaded_pisa)
    
    tab1, tab2 = st.tabs(["Interface Overview & 3D Viewer", "Detailed Interaction Analysis"])

    with tab1:
        st.header("Interface Statistics Summary")
        if not pisa_data['stats']:
            st.warning("No interface statistics found in the XML file.")
        else:
            stats_df = pd.DataFrame(pisa_data['stats'])
            stats_df_display = stats_df.rename(columns={
                'id': 'ID', 'int_area': 'Interface Area (Å²)', 'int_solv_en': 'ΔG Solv. (kcal/mol)',
                'stab_en': 'Stability Energy (kcal/mol)', 'pvalue': 'P-value', 'molecules': 'Interacting Molecules'
            })
            st.dataframe(stats_df_display, use_container_width=True, hide_index=True)

            st.markdown("---")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.header("Select Interface for Viewing")
                interface_ids = [s['id'] for s in pisa_data['stats']]
                if not interface_ids:
                    st.warning("No interfaces available for selection.")
                else:
                    selected_id = st.selectbox("Select Interface ID", options=interface_ids, format_func=lambda x: f"Interface {x}")
                    
                    selected_stats = next((item for item in pisa_data['stats'] if item['id'] == selected_id), None)
                    if selected_stats:
                        st.subheader(f"Statistics for Interface {selected_id}")
                        stat_map = {
                            'int_area': ("Interface Area (Å²)", "{:.2f} Å²"),
                            'int_solv_en': ("ΔG Solv. (kcal/mol)", "{:.2f}"),
                            'stab_en': ("Stability Energy (kcal/mol)", "{:.2f}"),
                            'pvalue': ("P-value", "{:.2e}")
                        }
                        for key, (label, fmt) in stat_map.items():
                            value = selected_stats.get(key)
                            if value is not None:
                                st.metric(label=label, value=fmt.format(value) if isinstance(value, float) else value)
                        
                        molecules_str = selected_stats.get('molecules')
                        if molecules_str:
                            st.markdown(f"**Interacting Molecules:**")
                            st.text(molecules_str)
            
            with col2:
                if uploaded_pdb:
                    st.header("3D Visualization")
                    residues_to_highlight = pisa_data['interfaces'].get(selected_id, {}).get('residues', {})
                    interacting_chains = set(residues_to_highlight.keys())
                    
                    if not interacting_chains:
                        st.warning(f"No interacting residues found for Interface {selected_id} to display.")
                    else:
                        # 1. Filter the PDB file to keep only interacting chains
                        uploaded_pdb.seek(0)
                        pdb_lines = uploaded_pdb.read().decode("utf-8").splitlines()
                        filtered_pdb_lines = [line for line in pdb_lines if (line.startswith("ATOM") or line.startswith("HETATM")) and line[21] in interacting_chains]
                        filtered_pdb_content = "\n".join(filtered_pdb_lines)

                        # 2. Save to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode='w') as tmp_file:
                            tmp_file.write(filtered_pdb_content)
                            pdb_file_path = tmp_file.name
                        
                        # 3. Display using streamlit-molstar
                        st_molstar(pdb_file_path, key=f"molstar_viewer_{selected_id}", height='600px')
                        
                        # 4. Clean up the temporary file
                        os.unlink(pdb_file_path)
                else:
                    st.info("Upload a PDB file to visualize the selected interface.")

    with tab2:
        st.header("Detailed Interaction Analysis")
        unique_chains = pisa_data['all_chains']
        all_bonds = [bond for iface in pisa_data['interfaces'].values() for bond in iface['bonds']]

        with st.expander("Chain Type Assignment (for display only)"):
            st.markdown("Assign a descriptive type for each chain.")
            df_chains = pd.DataFrame({'Chain ID': unique_chains, 'Chain Type': unique_chains})
            edited_df = st.data_editor(df_chains, hide_index=True, column_config={"Chain ID": st.column_config.TextColumn(disabled=True)})

        chain_id_to_label = dict(zip(edited_df['Chain ID'], edited_df['Chain Type']))
        has_custom_names = list(edited_df['Chain ID']) != list(edited_df['Chain Type'])

        st.markdown("---")
        st.markdown("### Select Interacting Groups")
        col_group1, col_group2 = st.columns(2)
        group1 = col_group1.multiselect("Select chains for Group 1", options=unique_chains)
        group2 = col_group2.multiselect("Select chains for Group 2", options=unique_chains)

        st.markdown("---")
        st.markdown("### ⚙️ Table Display Options")

        protein1_name, protein2_name = "Protein 1", "Protein 2"
        if has_custom_names:
            name_col1, name_col2 = st.columns(2)
            protein1_name = name_col1.text_input("Column header for Group 1", "Antigen")
            protein2_name = name_col2.text_input("Column header for Group 2", "Antibody")

        show_chain_id = st.checkbox("Show original Chain ID in residue columns", value=True)
        group_identical = st.checkbox("Group identical interactions across symmetric chains", value=False, help="If multiple identical chains make the same interaction, show it as a single row.")

        grouping_choice = st.radio(
            "Summarize interactions by:", options=['group1', 'group2', 'none'],
            format_func=lambda x: {
                'group1': f"Group 1 ({protein1_name})", 'group2': f"Group 2 ({protein2_name})",
                'none': "No summarization (show all pairs)"
            }[x],
            horizontal=True, help="Choose how to group interactions in the table."
        )

        if group1 and group2:
            st.markdown("---")
            st.code(find_residues_and_generate_chimerax_command(all_bonds, group1, group2), language="bash")
            table_rows = generate_interaction_table(all_bonds, group1, group2, chain_id_to_label, show_chain_id, grouping_choice, group_identical)
            
            if table_rows:
                st.subheader("Interaction Details Table")
                df_details = pd.DataFrame(table_rows, columns=["Protein 1", "Residue 1", "Protein 2", "Residue 2", "Interaction Type", "Atom Interaction"])
                
                def extract_res_num_for_sort(text):
                    first_residue = str(text).split(',')[0].strip()
                    try:
                        num_str = "".join(filter(str.isdigit, first_residue.split()[-1]))
                        return int(num_str) if num_str else 0
                    except (ValueError, IndexError): return 0

                df_details["Res #1"] = df_details["Residue 1"].apply(extract_res_num_for_sort)
                df_details["Res #2"] = df_details["Residue 2"].apply(extract_res_num_for_sort)
                
                primary_sort_num_col = "Res #2" if grouping_choice == 'group2' else "Res #1"
                secondary_sort_num_col = "Res #1" if grouping_choice == 'group2' else "Res #2"
                df_details = df_details.sort_values(by=[primary_sort_num_col, secondary_sort_num_col])

                if has_custom_names:
                    final_cols = [protein1_name, "Residue 1", "Res #1", protein2_name, "Residue 2", "Res #2", "Interaction Type", "Atom Interaction"]
                    df_details = df_details.rename(columns={"Protein 1": protein1_name, "Protein 2": protein2_name})
                else:
                    df_details = df_details.rename(columns={"Protein 1": "Selection 1", "Protein 2": "Selection 2"})
                    final_cols = ["Selection 1", "Res #1", "Selection 2", "Res #2", "Interaction Type", "Atom Interaction"]
                
                df_details = df_details[final_cols]
                st.dataframe(df_details, use_container_width=True, hide_index=True)
            else:
                st.warning("No interactions found for the selected groups.")
        else:
            st.info("Select chains for each group to see the interaction table.")

else:
    st.info("Awaiting PISA XML file upload.")