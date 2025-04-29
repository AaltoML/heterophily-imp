import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import re
import sys
# sys.path.insert(0,'..')
from hetflow.metrics import fraction_valid, remove_invalid, fraction_unique


atom_decoder_m = {0: 6, 1: 7, 2: 8, 3: 9}
bond_decoder_m = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
ATOM_VALENCY = {6:4, 7:3, 8:2, 9:1, 15:3, 16:2, 17:1, 35:1, 53:1}

from rdkit.Chem.Draw import IPythonConsole # This package is not used, but there will be weird bug in Draw.MolsToGridImage without this `import``.....
import cairosvg
import wandb

def molecules_draw_to_path(mol_list, path = None, wandb_option= False, num_row = 5):
    ''' generate images for mol_list, if path exist, save png there. If wandb_option True, return wanb_img'''
    if len(mol_list)>25:
        mol_list = mol_list[:25]
    # if legends is None:
    #     legends=[str(i) for i in range(len(mol_list))]
    img = Draw.MolsToGridImage(mol_list, molsPerRow=num_row, subImgSize=(300, 300),  returnPNG=False, useSVG=True)
    # img.show()
    # img.save(path)
    if path:
        cairosvg.svg2pdf(bytestring=img.data, write_to=path+".pdf")
    if wandb_option:
        img = Draw.MolsToGridImage(mol_list, molsPerRow=5, subImgSize=(300, 300), returnPNG=False, useSVG=False)
        # img.save("image.png")
        return wandb.Image(img)
    return img


def evaluation_smiles_for_debug(smiles):
    smiles_without_dot = [s if '.' not in s else None for s in smiles]
    metrics = dict()
    metrics['TEST/validity'] = fraction_valid(smiles_without_dot)
    metrics['TEST/validity_with_dot'] = fraction_valid(smiles)
    valid_smiles = remove_invalid(smiles_without_dot)
    if len(valid_smiles) ==0:
        return metrics, None
    metrics['TEST/uniquess'] = fraction_unique(valid_smiles)
    valid_mols = [Chem.MolFromSmiles(s) for s in valid_smiles]
    return metrics, valid_mols

def flatten_graph_data(adj, x):
    return torch.cat((adj.reshape([adj.shape[0], -1]), x.reshape([x.shape[0], -1])), dim=1)


def split_channel(x):
    n = x.shape[1] // 2
    return x[:, :n], x[:, n:]

def generate_mols_data(model, temp=0.7, z_mu=None, batch_size=20, true_adj=None, device=-1, layer_wise = False):  #  gpu=-1):
    """

    :param model: Moflow model
    :param z_mu: latent vector of a molecule
    :param batch_size:
    :param true_adj:
    :param gpu:
    :return:
    """
    # xp = np
    if isinstance(device, torch.device):
        # xp = chainer.backends.cuda.cupy
        pass
    elif isinstance(device, int):
        if device >= 0:
            # device = args.gpu
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', int(device))
        else:
            device = torch.device('cpu')
    else:
        raise ValueError("only 'torch.device' or 'int' are valid for 'device', but '%s' is "'given' % str(device))

    z_dim = model.b_size + model.a_size  # 324 + 45 = 369   9*9*4 + 9 * 5
    mu = np.zeros(z_dim)  # (369,) default , dtype=np.float64
    sigma_diag = np.ones(z_dim)  # (369,)

    if model.hyper_params.learn_dist:
        if len(model.ln_var) == 1:
            sigma_diag = np.sqrt(np.exp(model.ln_var.item())) * sigma_diag
        elif len(model.ln_var) == 2:
            sigma_diag[:model.b_size] = np.sqrt(np.exp(model.ln_var[0].item())) * sigma_diag[:model.b_size]
            sigma_diag[model.b_size+1:] = np.sqrt(np.exp(model.ln_var[1].item())) * sigma_diag[model.b_size+1:]

        # sigma_diag = xp.exp(xp.hstack((model.ln_var_x.data, model.ln_var_adj.data)))

    sigma = temp * sigma_diag

    with torch.no_grad():
        if z_mu is not None:
            mu = z_mu
            sigma = 0.01 * np.eye(z_dim)
        # mu: (369,), sigma: (369,), batch_size: 100, z_dim: 369
        z = np.random.normal(mu, sigma, (batch_size, z_dim))  # .astype(np.float32)
        z = torch.from_numpy(z).float().to(device)
        adj, x = model.reverse(z, true_adj=true_adj, layer_wise = layer_wise)

    return (adj, x)  # (bs, 4, 9, 9), (bs, 9, 5)



def generate_smiles_from_data(adj, x, atomic_num_list, mask = None, gpu=-1, return_unique=True, debug=True):
    """

    :param adj:  (100,4,9,9)
    :param x: (100.9,5)
    :param atomic_num_list: [6,7,8,9,0]
    :param gpu:  e.g. gpu0
    :param return_unique:
    :return:
    """
    adj = _to_numpy_array(adj)  # , gpu)  (1000,4,9,9)
    x = _to_numpy_array(x)  # , gpu)  (1000,9,5)
    if mask ==None:
        mols = [construct_mol(x_elem, adj_elem, atomic_num_list)
                for x_elem, adj_elem in zip(x, adj)]   #len()=1000
    else:
        mask =  _to_numpy_array(mask)
        mols = [construct_mol(x_elem[m], adj_elem, atomic_num_list)
                for x_elem, adj_elem, m in zip(x, adj, mask)]   #len()=1000
    smiles = [Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None for mol in mols]
    # valid = [mol for mol in smiles if mol is not None]
    return smiles


def get_graph_data(x, num_nodes, num_relations, num_features):
    """
    Converts a vector of shape [b, num_nodes, m] to Adjacency matrix
    of shape [b, num_relations, num_nodes, num_nodes]
    and a feature matrix of shape [b, num_nodes, num_features].
    :param x:
    :param num_nodes:
    :param num_relations:
    :param num_features:
    :return:
    """
    adj = x[:, :num_nodes*num_nodes*num_relations].reshape([-1, num_relations, num_nodes, num_nodes])
    feat_mat = x[:, num_nodes*num_nodes*num_relations:].reshape([-1, num_nodes, num_features])
    return adj, feat_mat


def Tensor2Mol(A, x):
    mol = Chem.RWMol()
    # x[x < 0] = 0.
    # A[A < 0] = -1
    # atoms_exist = np.sum(x, 1) != 0
    atoms = np.argmax(x, 1)
    atoms_exist = atoms != 4
    atoms = atoms[atoms_exist]
    atoms += 6
    adj = np.argmax(A, 0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    # print('num atoms: {}'.format(sum(atoms>0)))

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atom)))

    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])

    return mol


def construct_mol(x, A, atomic_num_list):
    """

    :param x:  (9,5)
    :param A:  (4,9,9)
    :param atomic_num_list: [6,7,8,9,0]
    :return:
    """
    mol = Chem.RWMol()
    # x (ch, num_node)
    atoms = np.argmax(x, axis=1)
    # last a
    atoms_exist = atoms != len(atomic_num_list) - 1
    atoms = atoms[atoms_exist]
    # print('num atoms: {}'.format(sum(atoms>0)))

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    # A (edge_type, num_node, num_node)
    adj = np.argmax(A, axis=0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])
            # add formal charge to atom: e.g. [O+], [N+] [S+]
            # not support [O-], [N-] [S-]  [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def construct_mol_with_validation(x, A, atomic_num_list):
    """

    :param x:  (9,5)
    :param A:  (4,9,9)
    :param atomic_num_list: [6,7,8,9,0]
    :return:
    """
    mol = Chem.RWMol()
    # x (ch, num_node)
    atoms = np.argmax(x, axis=1)
    # last a
    atoms_exist = atoms != len(atomic_num_list) - 1
    atoms = atoms[atoms_exist]
    # print('num atoms: {}'.format(sum(atoms>0)))

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    # A (edge_type, num_node, num_node)
    adj = np.argmax(A, axis=0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])
            t = adj[start, end]
            while not valid_mol_can_with_seg(mol):
                mol.RemoveBond(int(start), int(end))
                t = t-1
                if t >= 1:
                    mol.AddBond(int(start), int(end), bond_decoder_m[t])

    return mol


def valid_mol(x):
    s = Chem.MolFromSmiles(Chem.MolToSmiles(x, isomericSmiles=True)) if x is not None else None
    if s is not None and '.' not in Chem.MolToSmiles(s, isomericSmiles=True):
        return s
    return None


def valid_mol_can_with_seg(x, largest_connected_comp=True):
    # mol = None
    if x is None:
        return None
    sm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = Chem.MolFromSmiles(sm)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    return mol


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(x):
    xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = x
    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len (atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder_m[t])
                # if '.' in Chem.MolToSmiles(mol, isomericSmiles=True):
                #     print(tt)
                #     print(Chem.MolToSmiles(mol, isomericSmiles=True))

    return mol


def test_correct_mol():
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(7))
    mol.AddBond(0, 1, Chem.rdchem.BondType.DOUBLE)
    mol.AddBond(1, 2, Chem.rdchem.BondType.TRIPLE)
    mol.AddBond(0, 3, Chem.rdchem.BondType.TRIPLE)
    print(Chem.MolToSmiles(mol))  # C#C=C#N
    mol = correct_mol(mol)
    print(Chem.MolToSmiles(mol))  # C=C=C=N


def check_tensor(x):
    return valid_mol(Tensor2Mol(*x))


def adj_to_smiles(adj, x, atomic_num_list):
    # adj = _to_numpy_array(adj, gpu)
    # x = _to_numpy_array(x, gpu)
    valid = [Chem.MolToSmiles(construct_mol(x_elem, adj_elem, atomic_num_list), isomericSmiles=True)
             for x_elem, adj_elem in zip(x, adj)]
    return valid


def check_validity(adj, x, atomic_num_list, gpu=-1, return_unique=True,
                   correct_validity=True, largest_connected_comp=True, debug=True):
    """

    :param adj:  (100,4,9,9)
    :param x: (100.9,5)
    :param atomic_num_list: [6,7,8,9,0]
    :param gpu:  e.g. gpu0
    :param return_unique:
    :return:
    """
    adj = _to_numpy_array(adj)  # , gpu)  (1000,4,9,9)
    x = _to_numpy_array(x)  # , gpu)  (1000,9,5)
    if correct_validity:
        # valid = [valid_mol_can_with_seg(construct_mol_with_validation(x_elem, adj_elem, atomic_num_list)) # valid_mol_can_with_seg
        #          for x_elem, adj_elem in zip(x, adj)]
        valid = []
        for x_elem, adj_elem in zip(x, adj):
            mol = construct_mol(x_elem, adj_elem, atomic_num_list)
            # Chem.Kekulize(mol, clearAromaticFlags=True)
            cmol = correct_mol(mol)
            vcmol = valid_mol_can_with_seg(cmol, largest_connected_comp=largest_connected_comp)   #  valid_mol_can_with_seg(cmol)  # valid_mol(cmol)  # valid_mol_can_with_seg
            # Chem.Kekulize(vcmol, clearAromaticFlags=True)
            valid.append(vcmol)
    else:
        valid = [valid_mol(construct_mol(x_elem, adj_elem, atomic_num_list))
             for x_elem, adj_elem in zip(x, adj)]   #len()=1000
    valid = [mol for mol in valid if mol is not None]  #len()=valid number, say 794
    if debug:
        print("valid molecules: {}/{}".format(len(valid), adj.shape[0]))
        for i, mol in enumerate(valid):
            print("[{}] {}".format(i, Chem.MolToSmiles(mol, isomericSmiles=False)))

    n_mols = x.shape[0]
    valid_ratio = len(valid)/n_mols  # say 794/1000
    valid_smiles = [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in valid]
    unique_smiles = list(set(valid_smiles))  # unique valid, say 788
    unique_ratio = 0.
    if len(valid) > 0:
        unique_ratio = len(unique_smiles)/len(valid)  # say 788/794
    if return_unique:
        valid_smiles = unique_smiles
    valid_mols = [Chem.MolFromSmiles(s) for s in valid_smiles]
    abs_unique_ratio = len(unique_smiles)/n_mols
    if debug:
        print("valid: {:.3f}%, unique: {:.3f}%, abs unique: {:.3f}%".
            format(valid_ratio * 100, unique_ratio * 100, abs_unique_ratio * 100))
    results = dict()
    results['valid_mols'] = valid_mols
    results['valid_smiles'] = valid_smiles
    results['valid_ratio'] = valid_ratio*100
    results['unique_ratio'] = unique_ratio*100
    results['abs_unique_ratio'] = abs_unique_ratio * 100

    return results


def check_novelty(gen_smiles, train_smiles, n_generated_mols): # gen: say 788, train: 120803
    if len(gen_smiles) == 0:
        novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]  # [1]*45
        novel = len(gen_smiles) - sum(duplicates)  # 788-45=743
        novel_ratio = novel*100./len(gen_smiles)  # 743*100/788=94.289
        abs_novel_ratio = novel*100./n_generated_mols
    print("novelty: {:.3f}%, abs novelty: {:.3f}%".format(novel_ratio, abs_novel_ratio))
    return novel_ratio, abs_novel_ratio


def _to_numpy_array(a):  # , gpu=-1):
    if isinstance(a, torch.Tensor):
        a = a.cpu().detach().numpy()
    # if gpu >= 0:
    #     return cuda.to_cpu(a)
    elif isinstance(a, np.ndarray):
        # We do not use cuda np.ndarray in pytorch
        pass
    else:
        raise TypeError("a ({}) is not a torch.Tensor".format(type(a)))
    return a


def save_mol_png(mol, filepath, size=(600, 600)):
    Draw.MolToFile(mol, filepath, size=size)


if __name__ == '__main__':

    test_correct_mol()