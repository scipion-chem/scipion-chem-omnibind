import json
import pandas as pd
from omnibind.predict import load_model, predict_single
from omegaconf import OmegaConf

def load_compounds(path):
    df = pd.read_csv(path)
    # assume column "smiles"
    return df["smiles"].tolist()

def run_batch(config_path):
    # load config
    with open(config_path) as f:
        cfg_data = json.load(f)

    # setup OmniBind config
    cfg = OmegaConf.load(cfg_data["config"])
    cfg.model.type = cfg_data["model_type"]

    cfg.model.hid_dim = 256

    cfg.model.encoder_aa.hid_dim = 256
    cfg.model.encoder_sa.hid_dim = 256
    cfg.model.protdecoder.hid_dim = 256
    cfg.model.decoder.hid_dim = 256

    cfg.model.encoder_aa.n_layers = 2
    cfg.model.encoder_sa.n_layers = 2
    cfg.model.protdecoder.n_layers = 2
    cfg.model.decoder.n_layers = 5

    cfg.model.num_encoder_layers = 2
    cfg.model.num_decoder_layers = 5

    cfg.model.encoder_aa.n_head = 4
    cfg.model.encoder_sa.n_head = 4
    cfg.model.protdecoder.n_head = 4
    cfg.model.decoder.n_head = 4
    cfg.model.cafb.n_head = 4

    cfg.training.device = cfg_data["device"]

    model = load_model(cfg, cfg_data["checkpoint"])

    molecules = load_compounds(cfg_data["compounds_csv"])
    sequences = cfg_data["sequences"]

    results = []

    for prot_id, seq in sequences.items():
        print(f'Processing protein: {prot_id}')
        batch_smiles = molecules

        outs = []
        for smi in batch_smiles:
            outs.append(
                predict_single(
                    smiles=smi,
                    aa_sequence=seq,
                    #sa_sequence=None,
                    sa_sequence="A" * len(seq),
                    model=model,
                    cfg=cfg
                )
            )

        for smi, out in zip(batch_smiles, outs):
            results.append({
                "protein": prot_id,
                "smiles": smi,
                "pKi": out["predicted_ki"],
                "pKd": out["predicted_kd"],
                "pIC50": out["predicted_ic50"],
                "pEC50": out["predicted_ec50"]
            })

    df = pd.DataFrame(results)
    df.to_csv(cfg_data["output"], index=False)

    print(f"Saved results: {cfg_data['output']}")


if __name__ == "__main__":
    import sys
    run_batch(sys.argv[1])

