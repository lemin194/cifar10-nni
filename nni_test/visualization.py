import io
import graphviz
import matplotlib.pyplot as plt
from PIL import Image
import json
from model_search import Network
import operations


def plot_single_cell(arch_dict, cell_name):
	g = graphviz.Digraph(
		node_attr=dict(style='filled', shape='rect', align='center'),
		format='png'
	)
	g.body.extend(['rankdir=LR'])

	g.node('c_{k-2}', fillcolor='darkseagreen2')
	g.node('c_{k-1}', fillcolor='darkseagreen2')
	assert len(arch_dict) % 2 == 0

	
	last_node = 0
	for edge in arch_dict[cell_name]:
		from_ = edge['from']
		to_ = edge['to']
		last_node = max((last_node, int(from_), int(to_)))
	
	print(last_node)

	for i in range(2, last_node):
		g.node(str(i), fillcolor='lightblue')

	g.node('c_{k}', fillcolor='palegoldenrod')

	for edge in arch_dict[cell_name]:
		from_ = edge['from']
		to_ = edge['to']
		op = edge['op']
		if int(from_) == 0:
			u = 'c_{k-2}'
		elif int(from_) == 1:
			u = 'c_{k-1}'
		elif int(from_) == last_node:
			u = 'c_{k}'
		else:
			u = str(from_)

		if int(to_) == 0:
			v = 'c_{k-2}'
		elif int(to_) == 1:
			v = 'c_{k-1}'
		elif int(to_) == last_node:
			v = 'c_{k}'
		else:
			v = str(to_)
		g.edge(u, v, label=op, fillcolor='gray')

	g.attr(label=f'{cell_name.capitalize()} cell')

	image = Image.open(io.BytesIO(g.pipe()))
	return image

def plot_double_cells(arch_dict):
	image1 = plot_single_cell(arch_dict, 'normal')
	image2 = plot_single_cell(arch_dict, 'reduce')
	height_ratio = max(image1.size[1] / image1.size[0], image2.size[1] / image2.size[0])
	_, axs = plt.subplots(1, 2, figsize=(20, 10 * height_ratio))
	axs[0].imshow(image1)
	axs[1].imshow(image2)
	axs[0].axis('off')
	axs[1].axis('off')
	plt.show()
	



def plot_arch(model_path):
	search_space = Network().get_search_space()
	best_arch = json.load(open(model_path))


	normal_space = {
		k.split('/')[1] : search_space['normal'][k] for k in search_space['normal'].keys()
	}
	reduce_space = {
		k.split('/')[1] : search_space['reduce'][k] for k in search_space['reduce'].keys()
	}



	normal_cell = []
	reduce_cell = []
	primitives = operations.DARTS_PRIMITIVES
	for key in best_arch:
		cell_type, name = key.split('/')
		if cell_type == 'normal':
			input = str.join('_', name.split('_')[:2])
			if (input[:5] != 'input'): continue
			conn = normal_space[input][int(best_arch[key])]
			from_ = conn.split('/')[1].split('_')[2]
			to_ = conn.split('/')[1].split('_')[1]
			op_ = primitives[best_arch[key]]
			normal_cell.append(
				{
					'from': from_,
					'to' : to_,
					'op' : op_
				}
			)
		if cell_type == 'reduce':
			input = str.join('_', name.split('_')[:2])
			if (input[:5] != 'input'): continue
			conn = reduce_space[input][int(best_arch[key])]
			from_ = conn.split('/')[1].split('_')[2]
			to_ = conn.split('/')[1].split('_')[1]
			op_ = primitives[best_arch[key]]
			reduce_cell.append(
				{
					'from': from_,
					'to' : to_,
					'op' : op_
				}
			)


	arch = {'normal': normal_cell, 'reduce': reduce_cell}

	plot_double_cells(arch)



plot_arch('./checkpoint/5w6hrljf/best_arch.json')