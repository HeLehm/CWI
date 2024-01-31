from dash import Dash, html, dcc, Input, Output, callback
import json
import dash_cytoscape as cyto
from transformers import AutoTokenizer
from src.simplify.visualization import BeamSearchDataLogger



def add_info_to_graph(graph, tokenizer):
    """
    adds info of possible tokens and their porbability and theri old probability
    """

    for node, data in graph.nodes(data=True):
        senetnce_so_far = tokenizer.decode(data['input_ids'], skip_special_tokens=True)
        input_ids_with_next_token = [
            [
                *data['input_ids'],
                data['next_token_ids'][i]
            ]
            for i in range(len(data['next_token_ids']))
        ]
        input_with_next_tokens = [
            tokenizer.decode(input_ids, skip_special_tokens=True)
            for input_ids in input_ids_with_next_token
        ]

        next_tokens = [
            token_seq[len(senetnce_so_far):]
            for token_seq in input_with_next_tokens
        ]


        # info format:
        # (new_rank, old_rank, token, prob)

        # 1. find old rank and prob
        old_ranks = []
        old_probs = []

        for i, next_token in enumerate(data['next_token_ids']):
            for u, old_token in enumerate(data['prev_next_token_ids']):
                if next_token == old_token:
                    old_ranks.append(u)
                    old_probs.append(data['prev_probabilities'][i])
                    break

        assert len(old_ranks) == len(data['next_token_ids']),\
            f"old ranks not found for all tokens ({len(old_ranks), len(data['next_token_ids'])})"

        # 2. new rank is just the index
        new_ranks = list(range(len(data['next_token_ids'])))


        data['info'] = [
            (new_ranks[i], old_ranks[i], token, int(data['probabilities'][i]*1000)/ 10., int(old_probs[i]*1000)/ 10.)
            for i, token in enumerate(next_tokens)
        ]


    return graph



app = Dash(__name__)

data = json.load(open("data.json"))
tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
G = BeamSearchDataLogger.data_to_graph(data, tokenizer=tokenizer)[0]
G = add_info_to_graph(G, tokenizer=tokenizer)
            

def info_to_markdown_table(info):
    table_str = "| old rank | new rank | prob | old prob |\n| ---- | ---- | ---- | ---- |"
    for i, next_token in enumerate(info):
        table_str += f"\n|" + " | ".join([str(e) for e in next_token]) +  "|"
    return table_str


def gx_to_cy(gx):
    nodes = []
    edges = []
    for node in gx.nodes:
        nodes.append({'data': {'id': node, **gx.nodes[node]}})
    for edge in gx.edges:
        edges.append({'data': {'source': edge[0], 'target': edge[1], 'weight': int(gx.edges[edge]['weight']*1000)/ 10.}})
    return nodes + edges


# root = data[0][0]['id']
# roots = "[id = '{}']".format(root)
roots = [G.nodes[node]['id'] for node in G.nodes if G.in_degree(node) == 0]

roots_str = "[id = '{}']".format("'], [id = '".join(roots))

print("root: ", roots_str)

app.layout = html.Div([
    html.Div([
        cyto.Cytoscape(
        id='cytoscape',
        elements=gx_to_cy(G),
        layout={'name': 'breadthfirst', 'roots': roots_str},
        style={'width': '70%', 'height': '800px'},
        stylesheet=[
            {
                'selector': 'edge',
                'style': {
                    'label': 'data(weight)'
                }
            },
            {
                'selector': 'node',
                'style': {
                    'label': 'data(label)'
                }
            },
        ]
    ),
    dcc.Markdown(
        id='cytoscape-selectedNodeData-markdown',
        style= {
            'width': '30%',
            'height': '100%',
            'border': 'solid',
        }, # posituon to the right of prev node
    ),
    ], style={'display': 'flex'}),
])

@callback(Output('cytoscape-selectedNodeData-markdown', 'children'),
              Input('cytoscape', 'selectedNodeData'))
def displaySelectedNodeData(data_list):
    if data_list is None:
        return ""

    nodes_list = [
        '#### ' + data['label'] + "\n\n" + info_to_markdown_table(data['info'])+ "\n\n"
        for data in data_list
    ]

    return "\n\n".join(nodes_list)





app.run_server(debug=True)