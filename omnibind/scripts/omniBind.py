import json, os
import pandas as pd
from omnibind.predict import load_model, predict_single
from omegaconf import OmegaConf


def loadCompounds(path):
    df = pd.read_csv(path)
    return df["smiles"].tolist()


def load3diFasta(path):
    stitched3di = {}
    if not os.path.exists(path):
        print(f"Error: FASTA file not found at {path}")
        return stitched3di

    with open(path) as f:
        currentRootId = None
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Get the first part of the header
                fullId = line[1:].split()[0]

                # Split from the right once
                parts = fullId.rsplit('_', 1)

                if len(parts) > 1 and len(parts[1]) == 1 and parts[1].isalpha():
                    currentRootId = parts[0]
                else:
                    currentRootId = fullId

                if currentRootId not in stitched3di:
                    stitched3di[currentRootId] = ""
            else:
                if currentRootId:
                    stitched3di[currentRootId] += line

    return stitched3di


def runBatch(config_path):
    with open(config_path) as f:
        cfgData = json.load(f)

    cfg = OmegaConf.load(cfgData["config"])
    cfg.model.type = cfgData["model_type"]


    cfg.training.device = cfgData["device"]

    model = load_model(cfg, cfgData["checkpoint"])
    molecules = loadCompounds(cfgData["compounds_csv"])
    sequences = cfgData["sequences"]

    saSequences = load3diFasta(cfgData["ss_file"])

    results = []
    for protId, aaSeq in sequences.items():
        print(f'Processing protein: {protId}')

        saSeq = saSequences.get(protId)

        if saSeq is None:
            print(f"Warning: No 3Di sequence found for {protId}. Skipping.")
            continue

        if len(aaSeq) != len(saSeq):
            print(f"Error: Length mismatch for {protId}!")
            print(f"  AA Sequence Length: {len(aaSeq)}")
            print(f"  3Di Sequence Length: {len(saSeq)}")
            continue

        for smi in molecules:
            try:
                out = predict_single(
                    smiles=smi,
                    aa_sequence=aaSeq,
                    sa_sequence=saSeq,
                    model=model,
                    cfg=cfg
                )

                results.append({
                    "protein": protId,
                    "smiles": smi,
                    "pKi": out["predicted_ki"],
                    "pKd": out["predicted_kd"],
                    "pIC50": out["predicted_ic50"],
                    "pEC50": out["predicted_ec50"]
                })
            except Exception as e:
                print(f"Failed prediction for {protId} and {smi}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(cfgData["output"], index=False)
    print(f"Saved results: {cfgData['output']}")


if __name__ == "__main__":
    import sys

    runBatch(sys.argv[1])
