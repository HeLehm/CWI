import torch
from transformers import LogitsProcessor
from collections import defaultdict
from copy import deepcopy
import networkx as nx
from typing import List, Dict

class BeamSearchDataLogger(LogitsProcessor):

    @classmethod
    def data_to_graph(cls, data, tokenizer=None):
        graphs = []
        # for each batch element
        for element in data.values():
            # group by steps
            steps = defaultdict(list)
            for step in element:
                steps[step['step']].append(step)


            # tune into list nd order by steps
            
            # TODO: hanlde non consecutive steps
            steps = [steps[i] for i in range(len(steps))]

            # remove repeating steps
            # ie. same input_ids, & next_token_ids
            for step_idx, step in enumerate(steps):
                steps[step_idx] = remove_redundant_nodes(step)


            # create graph
            graph = nx.DiGraph()
            first_step = True
            for s_idx, step in enumerate(steps):
                for decisionpoint in step:
                    decisionpoint['id'] = f"{decisionpoint['step']}-{decisionpoint['beam_idx']}-{decisionpoint['batch_idx']}"
                    graph.add_node(decisionpoint['id'], **decisionpoint)
                    # find parent
                    if first_step:
                        continue
                    
                    for parent in steps[s_idx-1]:
                        if parent['input_ids'] == decisionpoint['input_ids'][:-1]:
                            # find weight by checking parent's next_token_ids
                            parent_next_token_ids = parent['next_token_ids']

                            weight = None
                            for i, next_token_id in enumerate(parent_next_token_ids):
                                if next_token_id == decisionpoint['input_ids'][-1]:
                                    weight = parent['probabilities'][i]
                                    break
                                
                            graph.add_edge(parent['id'], decisionpoint['id'], weight=weight)

                first_step = False

            # adding labels
            if tokenizer is None:
                for node, data in graph.nodes(data=True):
                    data['label'] = f"{data['input_ids']}"
                return graph

            for node, data in graph.nodes(data=True):
                senetnce_so_far = tokenizer.decode(data['input_ids'], skip_special_tokens=True)

                data['label'] = senetnce_so_far

            graphs.append(graph)

        return graphs


    def __init__(self,num_beams, eos_token_id ,top_k=5):
        super().__init__()
        self.top_k = top_k

        self.num_beams = num_beams
        self.eos_token_id = eos_token_id

        self._beam_search_data = None
        self._current_step = None
        self.reset()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Convert logits to probabilities using softmax
        probs = torch.nn.functional.softmax(scores, dim=-1)
        
        # Store the top k probabilities and their corresponding token ids for each beam
        top_k_probs, top_k_ids = probs.topk(k=self.top_k, dim=-1)
        
        # Convert to list and store in our data structure
        for beam_batch_idx, (ids, prob) in enumerate(zip(top_k_ids.tolist(), top_k_probs.tolist())):
            # if eos token in inpt_ids, then we are done with this beam
            if self.eos_token_id in input_ids[beam_batch_idx]:
                continue
            # Assuming input_ids is a single beam, we append to that beam's history
            beam_idx = beam_batch_idx % self.num_beams
            batch_idx = beam_batch_idx // self.num_beams

            self._beam_search_data[batch_idx].append({
                'step': self._current_step,
                'batch_idx': batch_idx,
                'beam_idx': beam_idx,
                'input_ids': input_ids[beam_batch_idx].tolist(),
                'next_token_ids': ids,
                'probabilities': prob,
            })
        self._current_step += 1
        
        # Return the original scores to not alter the beam search behavior
        return scores
    
    def get_data(self, reset=True):
        ret = deepcopy(dict(self._beam_search_data))
        if reset:
            self.reset()
        return ret
    
    def get_graph(self, reset=True, tokenizer=None) -> List[nx.DiGraph]:
        data = self.get_data(reset=reset)
        return self.data_to_graph(data, tokenizer=tokenizer)

    def reset(self):
        self._beam_search_data = defaultdict(list)
        self._current_step = 0


class TwoStepBeamSearchDataLogger(BeamSearchDataLogger):
    """
    Same as BeamSearchDataLogger but assumes 2 __call__ calls per step
    """

    def __init__(self, num_beams, eos_token_id, top_k=5):
        super().__init__(num_beams, eos_token_id, top_k)
        self._internal_step = 0

    def reset(self):
        super().reset()
        self._internal_step = 0
    

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):

        is_initial_step = self._internal_step % 2 == 0
        self._internal_step += 1

        if is_initial_step:
            return super().__call__(input_ids, scores)

        # Convert logits to probabilities using softmax
        probs = torch.nn.functional.softmax(scores, dim=-1)
        
        # Store the top k probabilities and their corresponding token ids for each beam
        top_k_probs, top_k_ids = probs.topk(k=self.top_k, dim=-1)
        
        # Convert to list and store in our data structure
        for beam_batch_idx, (ids, prob) in enumerate(zip(top_k_ids.tolist(), top_k_probs.tolist())):
            # if eos token in inpt_ids, then we are done with this beam
            if self.eos_token_id in input_ids[beam_batch_idx]:
                continue


            # Assuming input_ids is a single beam, we append to that beam's history
            beam_idx = beam_batch_idx % self.num_beams
            batch_idx = beam_batch_idx // self.num_beams

            # find with batch_idx, beam_idx and self._current_step - 1
            data_to_search = self._beam_search_data[batch_idx]
            # filter by step
            data_to_search = [item for item in data_to_search if item['step'] == self._current_step - 1]
            # filter by beam_idx
            data_to_search = [item for item in data_to_search if item['beam_idx'] == beam_idx]
            
            assert len(data_to_search) == 1, f"There should be only one item in data_to_search, but found {data_to_search}"
            data_to_update = data_to_search[0]

            # update data
            # key 'next_token_ids'
            # key 'probabilities'

            # keep old data
            data_to_update['prev_next_token_ids'] = data_to_update['next_token_ids']
            data_to_update['prev_probabilities'] = data_to_update['probabilities']
            # update with new data
            data_to_update['next_token_ids'] = ids
            data_to_update['probabilities'] = prob
        
        # Return the original scores to not alter the beam search behavior
        return scores


def remove_redundant_nodes(data):

    unique_data = []
    seen = set()

    for item in data:
        identifier = (tuple(item['input_ids']), tuple(item['next_token_ids']), item['step'])
        if identifier not in seen:
            seen.add(identifier)
            unique_data.append(item)

    return unique_data
