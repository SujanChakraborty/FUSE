"""Running M-NMF."""

from modularity_nmf import MNMF
from param_parser import parameter_parser
from calculation_helper import tab_printer

def create_and_run_model(args):
    """
    Run the M-NMF model and return the final node embeddings as a NumPy array.
    """
    tab_printer(args)  # prints parameters
    model = MNMF(args)
    model.optimize()
    
    # After optimization, return the embeddings as a NumPy array
    return model.optimal_node_representations.to_numpy()


if __name__ == "__main__":
    args = parameter_parser()
    create_and_run_model(args)
