Search.setIndex({"docnames": ["changes", "index", "matgl", "matgl.config", "matgl.dataloader", "matgl.dataloader.dataset", "matgl.graph", "matgl.graph.compute", "matgl.graph.converters", "matgl.layers", "matgl.layers.atom_ref", "matgl.layers.bond_expansion", "matgl.layers.core", "matgl.layers.cutoff_functions", "matgl.layers.embedding_block", "matgl.layers.graph_conv", "matgl.layers.readout_block", "matgl.layers.three_body", "matgl.models", "matgl.models.ase_interface", "matgl.models.m3gnet", "matgl.models.megnet", "matgl.models.potential", "matgl.utils", "matgl.utils.maths", "modules"], "filenames": ["changes.md", "index.md", "matgl.rst", "matgl.config.rst", "matgl.dataloader.rst", "matgl.dataloader.dataset.rst", "matgl.graph.rst", "matgl.graph.compute.rst", "matgl.graph.converters.rst", "matgl.layers.rst", "matgl.layers.atom_ref.rst", "matgl.layers.bond_expansion.rst", "matgl.layers.core.rst", "matgl.layers.cutoff_functions.rst", "matgl.layers.embedding_block.rst", "matgl.layers.graph_conv.rst", "matgl.layers.readout_block.rst", "matgl.layers.three_body.rst", "matgl.models.rst", "matgl.models.ase_interface.rst", "matgl.models.m3gnet.rst", "matgl.models.megnet.rst", "matgl.models.potential.rst", "matgl.utils.rst", "matgl.utils.maths.rst", "modules.rst"], "titles": ["Changelog", "Introduction", "matgl package", "matgl.config module", "matgl.dataloader package", "matgl.dataloader.dataset module", "matgl.graph package", "matgl.graph.compute module", "matgl.graph.converters module", "matgl.layers package", "matgl.layers.atom_ref module", "matgl.layers.bond_expansion module", "matgl.layers.core module", "matgl.layers.cutoff_functions module", "matgl.layers.embedding_block module", "matgl.layers.graph_conv module", "matgl.layers.readout_block module", "matgl.layers.three_body module", "matgl.models package", "matgl.models.ase_interface module", "matgl.models.m3gnet module", "matgl.models.megnet module", "matgl.models.potential module", "matgl.utils package", "matgl.utils.maths module", "matgl"], "terms": {"initi": [0, 1, 5, 11, 15, 24], "work": [0, 1], "version": 0, "http": [1, 24], "github": [1, 24], "com": 24, "materialsvirtuallab": [], "matgl": 1, "blob": [], "main": 20, "licens": [], "materi": [1, 2, 8], "graph": [1, 2, 5, 10, 12, 15, 16, 17, 20, 21, 22, 24, 25], "librari": [1, 24], "deep": 1, "learn": 1, "mathemat": 1, "ar": 1, "natur": 1, "represent": [1, 8, 20], "collect": 1, "atom": [1, 7, 8, 10, 16, 19, 24], "e": [1, 24], "g": [1, 7, 8, 10, 12, 16, 20, 22, 24], "molecul": [1, 8], "crystal": 1, "model": [1, 2, 24, 25], "have": 1, "been": 1, "shown": 1, "consist": 1, "deliv": 1, "except": 1, "perform": [1, 12, 15, 16, 17, 20, 24], "surrog": 1, "predict": [1, 10, 21], "properti": [1, 10, 12, 19, 20], "In": 1, "thi": [1, 2, 9, 12, 16, 17, 24], "repositori": 1, "we": [1, 19, 24], "reimplement": 1, "3": [1, 11, 20, 24], "bodi": [1, 7, 17], "network": [1, 2], "its": 1, "predecessor": 1, "megnet": [0, 1, 2, 5, 15, 18, 25], "us": [1, 3, 7, 10, 17, 19, 24], "dgl": [1, 5, 7, 8, 10, 15, 16, 17, 20, 21, 22, 24], "The": [1, 12, 16, 20, 24], "goal": 1, "improv": 1, "usabl": 1, "extens": [1, 10], "scalabl": 1, "origin": [1, 24], "were": [1, 19], "implement": [1, 2, 12, 15, 17, 18, 21, 23, 24], "tensorflow": [1, 3], "effort": 1, "collabor": 1, "between": [1, 3, 7, 24], "virtual": 1, "lab": 1, "intel": 1, "santiago": 1, "miret": 1, "marcel": 1, "nassar": 1, "carmelo": 1, "gonzal": 1, "feb": 1, "16": [1, 3], "2023": 1, "both": 1, "architectur": [1, 15], "complet": 1, "expect": 1, "bug": 1, "new": [1, 19], "neural": 1, "incorpor": 1, "interact": [1, 7, 17], "A": [1, 15, 22, 24], "kei": 1, "differ": [1, 10, 24], "prior": 1, "addit": 1, "coordin": 1, "lattic": 1, "matrix": [1, 10, 24], "which": [1, 19, 24], "necessari": 1, "obtain": 1, "tensori": 1, "quantiti": 1, "forc": [1, 5, 19, 22], "stress": [1, 5, 19, 22], "via": 1, "auto": 1, "differenti": 1, "As": 1, "framework": 1, "ha": [1, 24], "divers": 1, "applic": 1, "includ": [1, 5, 15, 17], "interatom": 1, "potenti": [1, 2, 13, 18, 19, 25], "develop": 1, "With": 1, "same": 1, "train": [1, 4, 5, 8, 10, 11, 12, 14, 15, 16, 17, 20, 21, 22, 24], "data": [1, 3, 5, 24], "similarli": 1, "state": [1, 5, 8, 10, 14, 15, 20, 21, 22, 24], "art": 1, "machin": 1, "ml": 1, "iap": 1, "howev": 1, "featur": [1, 8, 10, 11, 12, 14, 15, 16, 17, 21], "flexibl": 1, "scale": 1, "chemic": 1, "space": 1, "One": [1, 12], "accomplish": 1, "univers": 1, "can": [1, 19], "across": 1, "entir": 1, "period": 1, "tabl": 1, "element": [1, 8, 10, 24], "relax": [1, 19], "project": 1, "like": 1, "previou": [1, 19], "achiev": 1, "mani": 1, "case": 1, "accuraci": 1, "better": 1, "similar": 1, "other": [1, 8, 12, 24], "For": [1, 24], "detail": 1, "benchmark": 1, "pleas": 1, "public": 1, "section": 1, "api": 1, "document": 1, "avail": 1, "page": 1, "cite": 1, "follow": 1, "chen": 1, "c": 1, "ong": 1, "s": 1, "p": 1, "nat": 1, "comput": [1, 2, 6, 12, 16, 17, 25], "sci": 1, "2": [1, 20, 24], "718": 1, "728": 1, "2022": 1, "doi": 1, "org": [1, 24], "10": [1, 24], "1038": 1, "s43588": 1, "022": 1, "00349": 1, "ye": 1, "w": 1, "zuo": 1, "y": 1, "zheng": 1, "chem": 1, "mater": 1, "2019": 1, "31": 1, "9": 1, "3564": 1, "3572": 1, "1021": 1, "ac": 1, "chemmat": 1, "9b01294": 1, "wa": [1, 24], "primarili": 1, "support": 1, "fund": 1, "u": 1, "depart": 1, "energi": [1, 5, 10, 19, 22], "offic": 1, "scienc": [1, 2], "basic": 1, "engin": 1, "divis": 1, "under": 1, "contract": 1, "de": 1, "ac02": 1, "05": 1, "ch11231": 1, "program": 1, "kc23mp": 1, "expans": [1, 11, 17, 24], "supercomput": 1, "cluster": 1, "extrem": 1, "discoveri": 1, "environ": 1, "xsede": 1, "nation": 1, "foundat": 1, "grant": 1, "number": [1, 7, 10, 19, 24], "aci": 1, "1548562": 1, "dataload": [2, 25], "dataset": [2, 4, 25], "convert": [2, 5, 6, 25], "layer": [2, 25], "atom_ref": [2, 9, 25], "bond_expans": [2, 9, 25], "core": [2, 9, 20, 25], "cutoff_funct": [2, 9, 25], "embedding_block": [2, 9, 25], "graph_conv": [2, 9, 25], "readout_block": [2, 9, 25], "three_bodi": [2, 9, 25], "ase_interfac": [2, 18, 25], "m3gnet": [0, 2, 9, 13, 15, 16, 18, 19, 22, 25], "util": [2, 25], "math": [2, 23, 25], "config": [2, 25], "packag": 25, "subpackag": 25, "submodul": 25, "modul": 25, "content": 25, "type": [3, 10], "class": [3, 5, 8, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24], "datatyp": 3, "base": [3, 5, 7, 8, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 24], "object": [3, 8, 19, 20, 24], "numpi": 3, "choos": 3, "float16": 3, "float32": 3, "np_float": 3, "np_int": 3, "int32": 3, "classmethod": [3, 20], "set_dtyp": 3, "data_typ": 3, "str": [3, 5, 8, 10, 11, 12, 14, 15, 16, 19, 20, 21, 24], "none": [3, 5, 10, 12, 14, 15, 16, 19, 20, 21, 22, 24], "method": [3, 23], "set": [3, 8, 10, 11, 19], "arg": [3, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 19, 20, 22, 24], "32": 3, "torch_float": 3, "torch": [3, 7, 12, 13, 15, 16, 19, 20, 22, 24], "torch_int": 3, "set_global_dtyp": 3, "function": [3, 11, 12, 13, 15, 16, 17, 24], "return": [3, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24], "creat": [4, 5, 6], "dgldataset": [4, 5], "tool": [5, 8], "construct": [5, 8, 13], "grph": 5, "m3gnetdataset": 5, "structur": [5, 7, 8, 10, 19], "list": [5, 8, 10, 12, 15, 16, 19, 21, 24], "threebody_cutoff": [5, 7, 20], "float": [5, 7, 8, 11, 13, 15, 17, 19, 20, 21, 24], "name": 5, "has_cach": 5, "filenam": [5, 19], "dgl_graph": 5, "bin": 5, "bool": [5, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 24], "check": 5, "exist": 5, "file": [5, 19], "store": 5, "true": [5, 12, 15, 19, 20, 21, 22, 24], "load": [5, 19, 20], "filename_line_graph": 5, "dgl_line_graph": 5, "filename_graph_attr": 5, "graph_attr": [5, 15, 19, 21, 22], "pt": [5, 20], "attr": [5, 15, 20, 22], "process": [5, 19], "tupl": [5, 8, 10, 15, 20, 22], "pymatgen": [5, 8, 10], "save": [5, 19, 20], "megnetdataset": 5, "label": [5, 10], "label_nam": 5, "0": [5, 8, 11, 19, 20, 24], "final": [5, 11, 24], "5": [5, 8, 11, 20, 24], "num_cent": [5, 11, 24], "int": [5, 11, 12, 14, 15, 16, 17, 19, 20, 21, 24], "20": [5, 24], "width": [5, 11, 24], "mgldataload": 5, "train_data": 5, "subset": 5, "val_data": 5, "test_data": 5, "collate_fn": 5, "callabl": [5, 12], "batch_siz": 5, "num_work": 5, "use_ddp": 5, "fals": [5, 11, 12, 14, 15, 19, 20, 22, 24], "valid": 5, "test": 5, "manipul": 6, "variou": [7, 23, 24], "oper": [7, 9, 15, 24], "compute_3bodi": 7, "dglgraph": [7, 8, 10, 12, 15, 17, 20, 21, 22], "calcul": [7, 19, 24], "three": [7, 17], "indic": [7, 24], "from": [7, 8, 20, 24], "pair": [7, 11], "l_g": [7, 20, 22], "contain": [7, 8, 9, 10, 18], "inform": 7, "triple_bond_indic": 7, "np": [7, 10, 20], "ndarrai": [7, 10, 20], "bond": [7, 11, 14, 16, 17, 24], "form": 7, "n_triple_ij": 7, "angl": 7, "each": [7, 24], "n_triple_i": 7, "n_triple_": 7, "compute_pair_vector_and_dist": 7, "vector": [7, 12, 24], "distanc": [7, 11, 13, 24], "bond_vec": 7, "tensor": [7, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22, 24], "two": [7, 24], "bond_dist": [7, 11, 24], "src": 7, "node": [7, 14, 15, 16, 17, 20, 21, 24], "dst": 7, "compute_theta_and_phi": 7, "edg": [7, 14, 15, 17, 21, 24], "theta": 7, "phi": [7, 24], "cos_theta": 7, "triple_bond_length": 7, "create_line_graph": 7, "g_batch": 7, "batch": [7, 10, 20], "cutoff": [7, 8, 11, 13, 17, 20, 24], "code": 8, "pmg2graph": 8, "element_typ": [8, 20], "ASE": 8, "also": 8, "ad": 8, "get_graph_from_atom": 8, "heterograph": 8, "get": [8, 10, 24], "an": [8, 12, 24], "input": [8, 12, 15, 21, 24], "paramet": [8, 12, 15, 24], "state_attr": [8, 10, 14, 20], "get_graph_from_molecul": 8, "mol": 8, "get_graph_from_structur": 8, "get_element_list": 8, "train_structur": 8, "dictionari": [8, 10, 20], "cover": 8, "directori": [9, 20], "offset": 10, "atomref": 10, "property_offset": 10, "arrai": [10, 24], "total": [10, 19], "system": 10, "fit": 10, "structs_or_graph": 10, "element_list": 10, "ani": 10, "dtype": 10, "scalartyp": 10, "refer": 10, "valu": [10, 24], "forward": [10, 11, 12, 14, 15, 16, 17, 20, 21, 22, 24], "offset_per_graph": 10, "get_feature_matrix": 10, "num_structur": 10, "num_el": 10, "gener": [11, 14], "spheric": [11, 17, 24], "bessel": [11, 17, 24], "gaussian": [11, 24], "bondexpans": 11, "max_l": [11, 15, 17, 20, 24], "max_n": [11, 15, 17, 20, 24], "rbf_type": [11, 20], "sphericalbessel": [11, 20], "smooth": [11, 24], "100": 11, "devic": [11, 12, 14, 15, 16, 17, 20, 21, 24], "cpu": [11, 12, 14, 15, 16, 17, 20, 21, 24], "expand": [11, 24], "bond_basi": 11, "radial": [11, 24], "basi": [11, 17, 24], "multi": 12, "perceptron": 12, "mlp": [12, 16], "helper": 12, "edgeset2set": 12, "input_dim": 12, "n_iter": 12, "n_layer": 12, "set2set": [12, 16], "feat": 12, "defin": [12, 16, 17], "everi": [12, 16, 17], "call": [12, 16, 17], "hot": 12, "reset_paramet": [12, 24], "reiniti": [12, 24], "learnabl": 12, "gatedmlp": 12, "in_feat": [12, 16], "dim": [12, 15, 16, 24], "activate_last": 12, "use_bia": 12, "gate": [12, 16], "should": [12, 16, 17], "overridden": [12, 16, 17], "all": [12, 16, 17], "subclass": [12, 16, 17], "although": [12, 16, 17], "recip": [12, 16, 17], "pass": [12, 16, 17, 20], "need": [12, 16, 17], "within": [12, 16, 17], "one": [12, 16, 17], "instanc": [12, 16, 17], "afterward": [12, 16, 17], "instead": [12, 16, 17], "sinc": [12, 16, 17, 24], "former": [12, 16, 17], "take": [12, 16, 17], "care": [12, 16, 17], "run": [12, 16, 17, 19], "regist": [12, 16, 17], "hook": [12, 16, 17, 19], "while": [12, 16, 17], "latter": [12, 16, 17], "silent": [12, 16, 17], "ignor": [12, 16, 17], "them": [12, 16, 17], "activ": [12, 14, 15, 16, 20], "bias_last": 12, "depth": 12, "appli": 12, "turn": 12, "output": [12, 14, 15], "in_featur": 12, "last_linear": 12, "linear": 12, "last": 12, "out_featur": 12, "cosine_cutoff": 13, "r": [13, 24], "cosin": 13, "radiu": [13, 17, 24], "polynomial_cutoff": 13, "polynomi": 13, "embed": 14, "option": [14, 19], "attribut": [14, 15, 16, 21, 24], "embeddingblock": 14, "degree_rbf": 14, "nn": [14, 15], "num_node_feat": [14, 15, 20], "num_edge_feat": [14, 15, 20], "num_node_typ": [14, 20], "num_state_feat": [14, 15, 20], "include_st": [14, 15, 20], "num_state_typ": [14, 20], "state_embedding_dim": [14, 20], "block": [14, 15, 24], "node_attr": 14, "edge_attr": 14, "node_feat": [14, 15, 16, 17, 20, 21], "edge_feat": [14, 15, 17, 21], "state_feat": [14, 24], "convolut": 15, "gcl": 15, "m3gnetblock": 15, "degre": 15, "conv_hidden": [15, 21], "dropout": [15, 21], "compris": 15, "sequenc": 15, "updat": [15, 17, 20], "graph_feat": 15, "m3gnetgraphconv": 15, "edge_update_func": 15, "edge_weight_func": 15, "node_update_func": 15, "node_weight_func": 15, "attr_update_func": 15, "attr_update_": 15, "global": 15, "state_upd": 15, "state_featur": [15, 24], "edge_update_": 15, "edge_upd": 15, "static": [15, 24], "from_dim": 15, "edge_dim": 15, "node_dim": 15, "attr_dim": 15, "whether": [15, 24], "state_dim": 15, "nodul": 15, "node_update_": 15, "node_upd": 15, "megnetblock": 15, "act": [15, 21], "skip": 15, "todo": [15, 21], "add": [15, 21], "doc": [15, 21], "param": [15, 21], "megnetgraphconv": 15, "edge_func": 15, "node_func": 15, "attr_func": 15, "readout": 16, "reducereadout": 16, "op": 16, "mean": 16, "field": [16, 20], "reduc": 16, "lower": 16, "dimension": 16, "could": 16, "sum": [16, 19, 24], "up": 16, "etc": 16, "set2setreadout": 16, "num_step": 16, "num_lay": 16, "weightedreadout": 16, "num_target": [16, 20], "feed": 16, "atomic_prperti": 16, "weightedreadoutpair": 16, "averag": 16, "i": [16, 24], "j": [16, 24], "weight": [16, 20], "sphericalbesselwithharmon": 17, "use_smooth": [17, 20, 24], "use_phi": [17, 20, 24], "harmon": [17, 24], "line_graph": 17, "threebodyinteract": 17, "update_network_atom": 17, "update_network_bond": 17, "kwarg": [17, 19, 20], "3d": 17, "three_basi": 17, "three_cutoff": 17, "dynam": 19, "m3gnetcalcul": 19, "compute_stress": 19, "stress_weight": 19, "1": [19, 20, 24], "ase": 19, "system_chang": 19, "monitor": 19, "chang": 19, "If": 19, "result": [19, 24], "implemented_properti": 19, "free_energi": 19, "hessian": [19, 22], "handl": 19, "moleculardynam": 19, "ensembl": 19, "nvt": 19, "temperatur": 19, "300": 19, "timestep": 19, "pressur": 19, "6": 19, "324209121801212e": 19, "07": 19, "taut": 19, "taup": 19, "compressibility_au": 19, "trajectori": 19, "logfil": 19, "loginterv": 19, "append_trajectori": 19, "molecular": 19, "step": 19, "thin": 19, "wrapper": 19, "md": 19, "set_atom": 19, "optim": 19, "fire": 19, "relax_cel": 19, "01": 19, "fmax": 19, "500": 19, "traj_fil": 19, "interv": 19, "verbos": 19, "toler": 19, "converg": 19, "here": 19, "max": [19, 24], "trajectoryobserv": 19, "observ": 19, "intermedi": 19, "compute_energi": 19, "just": 19, "64": 20, "n_block": 20, "is_intens": 20, "readout_typ": 20, "weighted_atom": 20, "task_typ": 20, "regress": 20, "4": [20, 24], "unit": 20, "data_mean": 20, "data_std": 20, "num_s2s_step": 20, "num_s2s_lay": 20, "element_ref": 20, "swish": [20, 21], "as_dict": 20, "messag": 20, "line": 20, "ouput": 20, "from_dict": 20, "dict": 20, "build": 20, "from_dir": 20, "path": 20, "model_dir": 20, "mp": 20, "2021": 20, "8": 20, "ef": 20, "pre": 20, "default": [20, 24], "in_dim": 21, "num_block": 21, "hidden": 21, "s2s_num_lay": 21, "s2s_num_it": 21, "output_hidden": 21, "is_classif": 21, "node_emb": 21, "edge_emb": 21, "attr_emb": 21, "graph_transform": 21, "calc_forc": 22, "calc_stress": 22, "calc_hessian": 22, "cwd": 24, "user": 24, "shyue": 24, "repo": 24, "precomput": 24, "root": 24, "2d": 24, "dimens": 24, "128": 24, "n": 24, "th": 24, "index": 24, "order": 24, "l": 24, "entri": 24, "gaussianexpans": 24, "shape": 24, "m": 24, "where": 24, "center": 24, "sphericalbesselfunct": 24, "sympi": 24, "pytorch": 24, "rbf_j0": 24, "ensur": 24, "vanish": 24, "first": 24, "sphericalharmonicsfunct": 24, "broadcast": 24, "input_tensor": 24, "target_tensor": 24, "along": 24, "given": 24, "match": 24, "target": 24, "modifi": 24, "torch_scatt": 24, "rusty1": 24, "pytorch_scatt": 24, "whose": 24, "inout": 24, "after": 24, "broadcast_states_to_atom": 24, "ns": 24, "nstate": 24, "nb": 24, "broadcast_states_to_bond": 24, "combine_sbf_shf": 24, "sbf": 24, "shf": 24, "combin": 24, "column": 24, "becom": 24, "get_range_indices_from_n": 24, "give": 24, "rang": 24, "get_segment_indices_from_n": 24, "segment": 24, "exampl": 24, "repeat_with_n": 24, "repeat": 24, "accord": 24, "replic": 24, "scatter_sum": 24, "segment_id": 24, "num_seg": 24, "scatter": 24, "specifi": 24, "id": 24, "spherical_bessel_root": 24, "fact": 24, "j_l": 24, "z_": 24, "On": 24, "hand": 24, "know": 24, "precis": 24, "j0": 24, "x": 24, "size": 24, "spherical_bessel_smooth": 24, "orthogon": 24, "second": 24, "deriv": 24, "equal": 24, "zero": 24, "ref": 24, "arxiv": 24, "pdf": 24, "1907": 24, "02374": 24, "boundari": 24, "unsorted_segment_fract": 24, "fraction": 24}, "objects": {"": [[2, 0, 0, "-", "matgl"]], "matgl": [[3, 0, 0, "-", "config"], [4, 0, 0, "-", "dataloader"], [6, 0, 0, "-", "graph"], [9, 0, 0, "-", "layers"], [18, 0, 0, "-", "models"], [23, 0, 0, "-", "utils"]], "matgl.config": [[3, 1, 1, "", "DataType"], [3, 4, 1, "", "set_global_dtypes"]], "matgl.config.DataType": [[3, 2, 1, "", "np_float"], [3, 2, 1, "", "np_int"], [3, 3, 1, "", "set_dtype"], [3, 2, 1, "", "torch_float"], [3, 2, 1, "", "torch_int"]], "matgl.dataloader": [[5, 0, 0, "-", "dataset"]], "matgl.dataloader.dataset": [[5, 1, 1, "", "M3GNetDataset"], [5, 1, 1, "", "MEGNetDataset"], [5, 4, 1, "", "MGLDataLoader"]], "matgl.dataloader.dataset.M3GNetDataset": [[5, 3, 1, "", "has_cache"], [5, 3, 1, "", "load"], [5, 3, 1, "", "process"], [5, 3, 1, "", "save"]], "matgl.dataloader.dataset.MEGNetDataset": [[5, 3, 1, "", "has_cache"], [5, 3, 1, "", "load"], [5, 3, 1, "", "process"], [5, 3, 1, "", "save"]], "matgl.graph": [[7, 0, 0, "-", "compute"], [8, 0, 0, "-", "converters"]], "matgl.graph.compute": [[7, 4, 1, "", "compute_3body"], [7, 4, 1, "", "compute_pair_vector_and_distance"], [7, 4, 1, "", "compute_theta_and_phi"], [7, 4, 1, "", "create_line_graph"]], "matgl.graph.converters": [[8, 1, 1, "", "Pmg2Graph"], [8, 4, 1, "", "get_element_list"]], "matgl.graph.converters.Pmg2Graph": [[8, 3, 1, "", "get_graph_from_atoms"], [8, 3, 1, "", "get_graph_from_molecule"], [8, 3, 1, "", "get_graph_from_structure"]], "matgl.layers": [[10, 0, 0, "-", "atom_ref"], [11, 0, 0, "-", "bond_expansion"], [12, 0, 0, "-", "core"], [13, 0, 0, "-", "cutoff_functions"], [14, 0, 0, "-", "embedding_block"], [15, 0, 0, "-", "graph_conv"], [16, 0, 0, "-", "readout_block"], [17, 0, 0, "-", "three_body"]], "matgl.layers.atom_ref": [[10, 1, 1, "", "AtomRef"]], "matgl.layers.atom_ref.AtomRef": [[10, 3, 1, "", "fit"], [10, 3, 1, "", "forward"], [10, 3, 1, "", "get_feature_matrix"], [10, 2, 1, "", "training"]], "matgl.layers.bond_expansion": [[11, 1, 1, "", "BondExpansion"]], "matgl.layers.bond_expansion.BondExpansion": [[11, 3, 1, "", "forward"], [11, 2, 1, "", "training"]], "matgl.layers.core": [[12, 1, 1, "", "EdgeSet2Set"], [12, 1, 1, "", "GatedMLP"], [12, 1, 1, "", "MLP"]], "matgl.layers.core.EdgeSet2Set": [[12, 3, 1, "", "forward"], [12, 3, 1, "", "reset_parameters"], [12, 2, 1, "", "training"]], "matgl.layers.core.GatedMLP": [[12, 3, 1, "", "forward"], [12, 2, 1, "", "training"]], "matgl.layers.core.MLP": [[12, 5, 1, "", "depth"], [12, 3, 1, "", "forward"], [12, 5, 1, "", "in_features"], [12, 5, 1, "", "last_linear"], [12, 5, 1, "", "out_features"], [12, 2, 1, "", "training"]], "matgl.layers.cutoff_functions": [[13, 4, 1, "", "cosine_cutoff"], [13, 4, 1, "", "polynomial_cutoff"]], "matgl.layers.embedding_block": [[14, 1, 1, "", "EmbeddingBlock"]], "matgl.layers.embedding_block.EmbeddingBlock": [[14, 3, 1, "", "forward"], [14, 2, 1, "", "training"]], "matgl.layers.graph_conv": [[15, 1, 1, "", "M3GNetBlock"], [15, 1, 1, "", "M3GNetGraphConv"], [15, 1, 1, "", "MEGNetBlock"], [15, 1, 1, "", "MEGNetGraphConv"]], "matgl.layers.graph_conv.M3GNetBlock": [[15, 3, 1, "", "forward"], [15, 2, 1, "", "training"]], "matgl.layers.graph_conv.M3GNetGraphConv": [[15, 3, 1, "", "attr_update_"], [15, 3, 1, "", "edge_update_"], [15, 3, 1, "", "forward"], [15, 3, 1, "", "from_dims"], [15, 3, 1, "", "node_update_"], [15, 2, 1, "", "training"]], "matgl.layers.graph_conv.MEGNetBlock": [[15, 3, 1, "", "forward"], [15, 2, 1, "", "training"]], "matgl.layers.graph_conv.MEGNetGraphConv": [[15, 3, 1, "", "attr_update_"], [15, 3, 1, "", "edge_update_"], [15, 3, 1, "", "forward"], [15, 3, 1, "", "from_dims"], [15, 3, 1, "", "node_update_"], [15, 2, 1, "", "training"]], "matgl.layers.readout_block": [[16, 1, 1, "", "ReduceReadOut"], [16, 1, 1, "", "Set2SetReadOut"], [16, 1, 1, "", "WeightedReadOut"], [16, 1, 1, "", "WeightedReadOutPair"]], "matgl.layers.readout_block.ReduceReadOut": [[16, 3, 1, "", "forward"], [16, 2, 1, "", "training"]], "matgl.layers.readout_block.Set2SetReadOut": [[16, 3, 1, "", "forward"], [16, 2, 1, "", "training"]], "matgl.layers.readout_block.WeightedReadOut": [[16, 3, 1, "", "forward"], [16, 2, 1, "", "training"]], "matgl.layers.readout_block.WeightedReadOutPair": [[16, 3, 1, "", "forward"], [16, 2, 1, "", "training"]], "matgl.layers.three_body": [[17, 1, 1, "", "SphericalBesselWithHarmonics"], [17, 1, 1, "", "ThreeBodyInteractions"]], "matgl.layers.three_body.SphericalBesselWithHarmonics": [[17, 3, 1, "", "forward"], [17, 2, 1, "", "training"]], "matgl.layers.three_body.ThreeBodyInteractions": [[17, 3, 1, "", "forward"], [17, 2, 1, "", "training"]], "matgl.models": [[19, 0, 0, "-", "ase_interface"], [20, 0, 0, "-", "m3gnet"], [21, 0, 0, "-", "megnet"], [22, 0, 0, "-", "potential"]], "matgl.models.ase_interface": [[19, 1, 1, "", "M3GNetCalculator"], [19, 1, 1, "", "MolecularDynamics"], [19, 1, 1, "", "Relaxer"], [19, 1, 1, "", "TrajectoryObserver"]], "matgl.models.ase_interface.M3GNetCalculator": [[19, 3, 1, "", "calculate"], [19, 2, 1, "", "implemented_properties"]], "matgl.models.ase_interface.MolecularDynamics": [[19, 3, 1, "", "run"], [19, 3, 1, "", "set_atoms"]], "matgl.models.ase_interface.Relaxer": [[19, 3, 1, "", "relax"]], "matgl.models.ase_interface.TrajectoryObserver": [[19, 3, 1, "", "compute_energy"], [19, 3, 1, "", "save"]], "matgl.models.m3gnet": [[20, 1, 1, "", "M3GNet"]], "matgl.models.m3gnet.M3GNet": [[20, 3, 1, "", "as_dict"], [20, 3, 1, "", "forward"], [20, 3, 1, "", "from_dict"], [20, 3, 1, "", "from_dir"], [20, 3, 1, "", "load"], [20, 2, 1, "", "training"]], "matgl.models.megnet": [[21, 1, 1, "", "MEGNet"]], "matgl.models.megnet.MEGNet": [[21, 3, 1, "", "forward"], [21, 2, 1, "", "training"]], "matgl.models.potential": [[22, 1, 1, "", "Potential"]], "matgl.models.potential.Potential": [[22, 3, 1, "", "forward"], [22, 2, 1, "", "training"]], "matgl.utils": [[24, 0, 0, "-", "maths"]], "matgl.utils.maths": [[24, 6, 1, "", "CWD"], [24, 1, 1, "", "GaussianExpansion"], [24, 1, 1, "", "SphericalBesselFunction"], [24, 1, 1, "", "SphericalHarmonicsFunction"], [24, 4, 1, "", "broadcast"], [24, 4, 1, "", "broadcast_states_to_atoms"], [24, 4, 1, "", "broadcast_states_to_bonds"], [24, 4, 1, "", "combine_sbf_shf"], [24, 4, 1, "", "get_range_indices_from_n"], [24, 4, 1, "", "get_segment_indices_from_n"], [24, 4, 1, "", "repeat_with_n"], [24, 4, 1, "", "scatter_sum"], [24, 4, 1, "", "spherical_bessel_roots"], [24, 4, 1, "", "spherical_bessel_smooth"], [24, 4, 1, "", "unsorted_segment_fraction"]], "matgl.utils.maths.GaussianExpansion": [[24, 3, 1, "", "forward"], [24, 3, 1, "", "reset_parameters"], [24, 2, 1, "", "training"]], "matgl.utils.maths.SphericalBesselFunction": [[24, 3, 1, "", "rbf_j0"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:attribute", "3": "py:method", "4": "py:function", "5": "py:property", "6": "py:data"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "attribute", "Python attribute"], "3": ["py", "method", "Python method"], "4": ["py", "function", "Python function"], "5": ["py", "property", "Python property"], "6": ["py", "data", "Python data"]}, "titleterms": {"changelog": 0, "v0": 0, "1": 0, "0": 0, "introduct": 1, "statu": 1, "m3gnet": [1, 20], "refer": 1, "acknowledg": 1, "matgl": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], "packag": [2, 4, 6, 9, 18, 23], "subpackag": 2, "submodul": [2, 4, 6, 9, 18, 23], "modul": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], "content": [2, 4, 6, 9, 18, 23], "config": 3, "dataload": [4, 5], "dataset": 5, "graph": [6, 7, 8], "comput": 7, "convert": 8, "layer": [9, 10, 11, 12, 13, 14, 15, 16, 17], "atom_ref": 10, "bond_expans": 11, "core": 12, "cutoff_funct": 13, "embedding_block": 14, "graph_conv": 15, "readout_block": 16, "three_bodi": 17, "model": [18, 19, 20, 21, 22], "ase_interfac": 19, "megnet": 21, "potenti": 22, "util": [23, 24], "math": 24}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 56}})