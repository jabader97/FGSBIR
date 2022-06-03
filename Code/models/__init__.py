from models.baseline import FGSBIR_Model


def get_model(hp):
    if "baseline" in hp.model:
        return FGSBIR_Model(hp)
    else:
        raise ValueError("Please specify a valid model.")
