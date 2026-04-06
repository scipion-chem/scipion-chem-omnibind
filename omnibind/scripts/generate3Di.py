import os
import subprocess
import sys
import json


def runFoldseek(structDir, outputFasta):
    cmd = (
        f"foldseek createdb {structDir} protDB && "
        f"foldseek lndb protDB_h protDB_ss_h && "
        f"foldseek convert2fasta protDB_ss {outputFasta}"
    )

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        raise RuntimeError("Foldseek failed")


def main(config_path):
    with open(config_path) as f:
        cfg = json.load(f)

    structDir = cfg["structures_dir"]
    outputFasta = cfg["outputFasta"]

    if not os.path.exists(structDir):
        raise Exception(f"Structures directory not found: {structDir}")

    runFoldseek(structDir, outputFasta)

    print(f"3Di FASTA generated at: {outputFasta}")


if __name__ == "__main__":
    main(sys.argv[1])

