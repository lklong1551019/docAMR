import penman
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 1. Your AMR String
amr_string = """
(d / document
    :snt1 (s1.g / go-02
                :ARG0 (s1.p / person
                            :name (s1.n / name
                                        :op1 "Hailey"))
                :ARG4 (s1.c / city
                            :name (s1.n2 / name
                                         :op1 "London"))
                :time (s1.t / tomorrow))
    :snt2 (s2.p / plan-01
                :ARG0 (s2.s / she)
                :ARG1 (s2.g / go-02
                            :ARG0 s2.s
                            :ARG4 (s2.c2 / country
                                         :name (s2.n / name
                                                     :op1 "Italy"))
                            :time (s2.a / after
                                        :op1 (s2.c / city
                                                   :name (s2.n2 / name
                                                               :op1 "London")
                                                   :same-as s1.c))))
    :snt3 (s3.s / see-01
                :ARG0 (s3.s2 / she
                             :same-as s2.s)
                :ARG1 (s3.b / building
                            :name (s3.n / name
                                        :op1 "Big"
                                        :op2 "Ben")))
    :snt4 (s4.m / meet-03
                :ARG0 (s4.p / person
                            :name (s4.n / name
                                        :op1 "Phil")
                            :ARG0-of (s4.h / have-rel-role-91
                                           :ARG1 (s4.s / she
                                                       :same-as s2.s)
                                           :ARG2 (s4.f / friend)))
                :ARG1 s4.s
                :location (s4.c / city
                                :name (s4.n2 / name
                                             :op1 "London")
                                :same-as s1.c)))
"""

g = penman.decode(amr_string)
G = nx.DiGraph()

# 2. Logic to group nodes by sentence
sentence_colors = {
    'd': '#FFD700',      # Gold for Document Root
    's1': '#ADD8E6',     # Light Blue
    's2': '#90EE90',     # Light Green
    's3': '#FFB6C1',     # Light Pink
    's4': '#E6E6FA'      # Lavender
}

node_colors = []
for source, role, target in g.instances():
    G.add_node(source, label=target)
    # Determine color based on prefix (e.g., 's1' from 's1.g')
    prefix = source.split('.')[0] if '.' in source else source
    node_colors.append(sentence_colors.get(prefix, '#D3D3D3'))

# 3. Edge Styling
edge_list = []
edge_colors = []
edge_widths = []

for source, role, target in g.edges():
    G.add_edge(source, target, label=role)
    if role == ':same-as':
        edge_colors.append('#006400') # Dark Green for coref
        edge_widths.append(2.5)
    elif role.startswith(':snt'):
        edge_colors.append('#000000') # Solid black for sentence roots
        edge_widths.append(1.5)
    else:
        edge_colors.append('#A9A9A9') # Dark Gray for standard relations
        edge_widths.append(0.8)

# 4. Hierarchical Layout (Requires 'pygraphviz' or 'pydot')
# If you don't have these, use nx.spring_layout(G, k=1.5)
try:
    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout(G, prog='dot')
except ImportError:
    print("Graphviz not found. Falling back to Spring Layout.")
    pos = nx.spring_layout(G, k=2.0, iterations=50)

# 5. Drawing
plt.figure(figsize=(20, 12))

# Draw Nodes with slight border
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, 
                       edgecolors='black', linewidths=1, alpha=0.9)

# Labels (Node Concepts)
node_labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold')

# Edges with specific styles
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, 
                       arrowsize=20, connectionstyle='arc3,rad=0.1')

# Edge Labels (Roles)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, 
                             label_pos=0.6, alpha=0.8)

plt.title("docAMR: Multi-Sentence Semantic Graph (Sentence-Color Coded)", fontsize=16)
plt.axis('off')
plt.show()