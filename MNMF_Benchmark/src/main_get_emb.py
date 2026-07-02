"""Running M-NMF."""

from modularity_nmf_get_emb import MNMF
from param_parser import parameter_parser
from calculation_helper import tab_printer

def create_and_run_model(args):
    """
    Run M-NMF and return embeddings as a NumPy array
    """
    tab_printer(args)
    model = MNMF(args)
    embeddings_np = model.optimize()  # <-- now optimize() returns embeddings
    return embeddings_np



if __name__ == "__main__":
    args = parameter_parser()
    create_and_run_model(args)
