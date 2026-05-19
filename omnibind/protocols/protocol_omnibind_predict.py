# **************************************************************************
# *
# * Authors:     Blanca Pueche (blanca.pueche@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
import csv, json, os, shutil
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from pwchem.objects import  SetOfAtomStructsChem

from pwem.protocols import EMProtocol
from pyworkflow.protocol import params

from pwem.convert import cifToPdb
from pyworkflow.object import String, Float, Pointer

from pwchem import Plugin as pwchemPlugin
from pwchem.constants import OPENBABEL_DIC
from pwchem.utils import getBaseName
from pwchem.objects import SetOfAtomStructsChem

from .. import Plugin as omnibindPlugin
from ..constants import OMNIBIND_DIC

class ProtOmniBindPrediction(EMProtocol):
  """
  AI Generated:

  Protocol to use OmniBind.

    AI Generated:

        ProtOmniBindPrediction - User Manual

        Overview
        --------
        This protocol predicts binding affinities between protein structures and
        small molecules using the OmniBind deep learning framework. It supports
        both single structures and sets of structures, automatically extracting
        sequences and generating structural embeddings (3Di) required by the model.

        Inputs
        ------
        - **inputStructure / inputStructures**: Protein structure(s) (PDB/mmCIF) used
          for virtual screening.
        - **inputSmallMols**: SetOfSmallMolecules containing ligands to evaluate.
        - **inputLibrary**: Optional SmallMoleculesLibrary (SMILES-based input).
        - **useLibrary**: Whether to use a SMILES library instead of molecular files.
        - **useGpu**: Enables GPU acceleration for prediction.

        Workflow
        --------
        1. **SMILES preparation**:
           - Converts input molecules into SMILES format (if not using a library).
           - Stores mappings between molecule names and SMILES strings.

        2. **3Di generation**:
           - Extracts protein structures and generates 3Di structural sequences
             using Foldseek-based processing.
           - Outputs a FASTA file with structural encodings.

        3. **Sequence extraction**:
           - Parses input PDB/mmCIF files.
           - Extracts amino acid sequences from all valid chains.

        4. **OmniBind execution**:
           - Prepares a configuration file including compounds, sequences, and 3Di data.
           - Runs the OmniBind model using CPU or GPU.
           - Produces predictions for each protein?ligand pair.

        5. **Output generation**:
           - Parses prediction results (pKi, pKd, pIC50, pEC50).
           - Assigns scores to each molecule for each structure.
           - Outputs either:
             - A SetOfSmallMolecules annotated with scores, or
             - A combined dataset for multiple structures.

        Outputs
        -------
        - **SetOfSmallMolecules**: Molecules annotated with predicted binding scores
          for each protein structure.
        - **results.csv**: CSV file containing predicted affinities for all
          protein?ligand combinations.

        Practical Recommendations
        -------------------------
        - Use when structural information is available for target proteins.
        - Suitable for structure-based virtual screening workflows.
        - Ensure input structures are clean and biologically meaningful.

        Summary & Interpretation
        ------------------------
        - Predictions include:
            - pKi
            - pKd
            - pIC50
            - pEC50
        - Higher values indicate stronger predicted binding affinity.
        - Results can guide prioritization of compounds for docking or experiments.

        Warnings
        --------
        - Incorrect or low-quality structures may lead to unreliable predictions.
        - Sequence extraction may fail for non-standard residues or malformed files.
        - Large datasets may require significant computational resources.
        - Ensure OmniBind environment and checkpoints are correctly installed.
  """
  _label = 'omnibind virtual screening'

  def __init__(self, **kwargs):
    EMProtocol.__init__(self, **kwargs)
    self.stepsExecutionMode = params.STEPS_PARALLEL

  def _defineParams(self, form):
    form.addSection(label='Input')

    form.addHidden('useGpu', params.BooleanParam, default=True,
                   label="Use GPU for execution",
                   help="This protocol has both CPU and GPU implementation. Choose one.")

    form.addHidden('gpuList', params.StringParam, default='0',
                   label="Choose GPU IDs",
                   help="Comma-separated GPU devices that can be used.")

    iGroup = form.addGroup('Input')
    iGroup.addParam('input', params.EnumParam, label='Input structure(s) as: ', default=0,
                    choices=['AtomStruct', 'SetOfAtomStructs'],
                    help='How to input the input structure(s)')
    iGroup.addParam('inputStructure', params.PointerParam, pointerClass="AtomStruct", condition='input==0',
                    label='Input structure: ',
                    help='Protein structure to perform the screening on')
    iGroup.addParam('inputStructures', params.PointerParam, pointerClass="SetOfAtomStructs", condition='input==1',
                    label='Input structures: ',
                    help='Set of protein structures to perform the screening on')

    iGroup.addParam('useLibrary', params.BooleanParam, label='Use library as input : ', default=False,
                    help='Whether to use a SMI library SmallMoleculesLibrary object as input')

    iGroup.addParam('inputLibrary', params.PointerParam, pointerClass="SmallMoleculesLibrary",
                    label='Input library: ', condition='useLibrary',
                    help="Input Small molecules library to predict")
    iGroup.addParam('inputSmallMols', params.PointerParam, pointerClass="SetOfSmallMolecules",
                    label='Input small molecules: ', condition='not useLibrary',
                    help='Set of small molecules to input the model for predicting their interactions')


  def _insertAllSteps(self):
    if not self.useLibrary.get():
      self._insertFunctionStep(self.convertStep)
    self._insertFunctionStep(self.generate3DiStep)
    self._insertFunctionStep(self.predictStep)
    self._insertFunctionStep(self.createOutputStep)


  def convertStep(self):
    smiDir = self.getInputSMIDir()
    if not os.path.exists(smiDir):
      os.makedirs(smiDir)

    molDir = self.copyInputMolsInDir()
    args = ' --multiFiles -iD "{}" --pattern "{}" -of smi --outputDir "{}"'. \
      format(molDir, '*', smiDir)
    pwchemPlugin.runScript(self, 'obabel_IO.py', args, env=OPENBABEL_DIC, cwd=smiDir)

  def generate3DiStep(self):
      structDir = os.path.abspath(self._getExtraPath('structures'))
      os.makedirs(structDir, exist_ok=True)

      for struct in self._getInpStructs():
          src = struct.getFileName()
          bName = getBaseName(src)
          dst = os.path.join(structDir, bName + os.path.splitext(src)[1])
          if not os.path.exists(dst):
              shutil.copy(src, dst)

      ssFile = os.path.abspath(self._getExtraPath('proteins_3di.fasta'))

      config = {
          "structures_dir": structDir,
          "output_fasta": ssFile
      }

      configFile = os.path.abspath(self._getExtraPath('foldseek_config.json'))
      with open(configFile, "w") as f:
          json.dump(config, f, indent=2)

      protocolDir = os.path.dirname(__file__)
      scriptPath = os.path.abspath(
          os.path.join(protocolDir, "..", "scripts", "generate3Di.py")
      )

      omnibindPlugin.runCondaCommand(
          self,
          program="python",
          args=f"{scriptPath} {configFile}",
          condaDic=OMNIBIND_DIC,
          cwd=self._getExtraPath()
      )

  def predictStep(self):
    smisDic = self.getInputSMIs()
    protSeqsDic = self.getInputSeqsFromStructures()

    compoundsFile = os.path.abspath(self._getExtraPath('compounds.csv'))
    with open(compoundsFile, 'w') as f:
        f.write("smiles,id\n")
        for name, smi in smisDic.items():
            f.write(f"{smi},{name}\n")

    extraPath = os.path.abspath(self._getExtraPath())
    configFile = os.path.join(extraPath, "config.json")
    if self.useGpu.get():
        device = 'cuda'
    else:
        device = 'cpu'

    config = {
        "checkpoint": os.path.join(omnibindPlugin.getVar(OMNIBIND_DIC['home']), "OmniBind/checkpoints/application.pth"),
        "model_type": "aa3di_gmf",
        "compounds_csv": compoundsFile,
        "config": os.path.join(omnibindPlugin.getVar(OMNIBIND_DIC['home']), "OmniBind/configs/default.yaml"),
        "output": os.path.abspath(os.path.join(self.getPath(), "results.csv")),
        "sequences": protSeqsDic,
        "ss_file": os.path.abspath(self._getExtraPath('proteins_3di.fasta')),
        "device": device
    }
    with open(configFile, "w") as f:
        json.dump(config, f, indent=2)

    protocolDir = os.path.dirname(__file__)
    scriptPath = os.path.abspath(
        os.path.join(protocolDir, "..", "scripts", "omniBind.py")
    )

    omnibindPlugin.runCondaCommand(
        self,
        program="python",
        args=f"{scriptPath} {configFile}",
        condaDic=OMNIBIND_DIC,
        cwd=extraPath
    )

  def parseScoreDic(self, resultsFile, scoreNames):
      smisDic = self.getInputSMIs()
      smiToName = {v.strip(): k for k, v in smisDic.items()}

      intDic, data = {}, {}
      with open(resultsFile, 'r') as f:
          reader = csv.DictReader(f, delimiter=',')
          for row in reader:
              protID = row['protein'].strip()
              smi = row['smiles'].strip()
              molName = smiToName.get(smi)

              if not molName:
                  continue

              scoresDic = {sName: float(row[sName.split('_')[1]]) for sName in scoreNames}

              if molName not in intDic:
                  intDic[molName] = {}
              intDic[molName][protID] = scoresDic

              if protID not in data:
                  data[protID] = {}
              data[protID][molName] = scoresDic

      return intDic, data


  def createOutputStep(self):
      resultsFile = self.getPath("results.csv")
      scoreNames = ['OmniBind_pKi', 'OmniBind_pKd', 'OmniBind_pIC50', 'OmniBind_pEC50']
      intDic, data = self.parseScoreDic(resultsFile, scoreNames)

      inStructs = self._getInpStructs()
      outStructs = SetOfAtomStructsChem().create(outputPath=self._getPath())
      if self.input.get() == 1:
          outStructs.copyInfo(self.inputStructures.get())

      scoresJsonFile = self._getExtraPath('scoresFile.json')
      if hasattr(self.inputStructures.get(), 'getInteractScoresFile'):
        prevFile = self.inputStructures.get().getInteractScoresFile()
      else:
          prevFile = None
      if prevFile and os.path.exists(prevFile):
          shutil.copy(prevFile, scoresJsonFile)
      outStructs.setInteractScoresFile(scoresJsonFile)

      outStructs.setInteractScoresDic(data)

      outMols = self.inputLibrary.get() if self.useLibrary.get() else self.inputSmallMols.get()
      outStructs.setInteractMols(outMols)

      proteinIDs = []
      for struct in inStructs:
          outStruct = struct.clone()
          outStructs.append(outStruct)
          proteinIDs.append(getBaseName(struct.getFileName()))

      self._defineOutputs(outputAtomStructs=outStructs)

      if not self.useLibrary.get():
          inMols = self.inputSmallMols.get()
          outputSmallMols = inMols.createCopy(self._getPath(), copyInfo=True)

          for mol in inMols:
              nMol = mol.clone()
              molName = nMol.getMolName()

              molDic = intDic[molName]
              if len(inStructs) == 1:
                  sDic = molDic.get(proteinIDs[0], {})
                  for sName, val in sDic.items():
                      setattr(nMol, sName, Float(val))
              else:
                  for pID, sDic in molDic.items():
                      for sName, val in sDic.items():
                          setattr(nMol, f"{sName}_{pID}", Float(val))

              outputSmallMols.append(nMol)

          outputSmallMols.updateMolClass()
          self._defineOutputs(outputSmallMolecules=outputSmallMols)

      else:
          inLib = self.inputLibrary.get()
          mapDic = inLib.getLibraryMap(inverted=True, fullLine=True)
          oLibFile = self._getPath('outputLibrary.smi')

          with open(oLibFile, 'w') as f:
              if len(inStructs) == 1:
                  headers = scoreNames
                  for molName, molDic in intDic.items():
                      sDic = molDic.get(proteinIDs[0], {})
                      scoreStr = '\t'.join([str(sDic[sName]) for sName in scoreNames])
                      f.write(f'{mapDic[molName]}\t{scoreStr}\n')
              else:
                  headers = [f'{sName}_{pID}' for pID in proteinIDs for sName in scoreNames]
                  for molName, molDic in intDic.items():
                      newCols = []
                      for protID, sDic in molDic.items():
                          newCols += [str(sDic[sName]) for sName in scoreNames]

                      newStr = '\t'.join(newCols)
                      f.write(f'{mapDic[molName]}\t{newStr}\n')

          outputLib = inLib.clone()
          outputLib.setFileName(oLibFile)
          outputLib.setHeaders(inLib.getHeaders() + headers)
          self._defineOutputs(outputLibrary=outputLib)


  ############## UTILS ########################

  def copyInputMolsInDir(self):
    oDir = os.path.abspath(self._getTmpPath('inMols'))
    if not os.path.exists(oDir):
      os.makedirs(oDir)

    for mol in self.inputSmallMols.get():
      os.link(mol.getFileName(), os.path.join(oDir, os.path.split(mol.getFileName())[-1]))
    return oDir

  def getInputSMIDir(self):
    return os.path.abspath(self._getExtraPath('inputSMI'))

  def getInputSMIs(self):
    '''Return the smi mapping dictionary {smiName: smi}
    '''
    smisDic = {}
    if not self.useLibrary.get():
      iDir = self.getInputSMIDir()
      for file in os.listdir(iDir):
        with open(os.path.join(iDir, file)) as f:
          smi, title = f.readline().split()
          smisDic[title] = smi.strip()
    else:
      inLib = self.inputLibrary.get()
      smisDic = inLib.getLibraryMap(inverted=True)

    return smisDic

  def _getInpStructs(self):
      if self.input.get() == 0: # AtomStruct
          return [self.inputStructure.get()]
      return self.inputStructures.get()

  def getInputSeqsFromStructures(self):
      seqsDic = {}

      parser = PDBParser(QUIET=True)

      for i, struct in enumerate(self._getInpStructs()):
          filePath = struct.getFileName()

          if filePath.endswith(".cif") or filePath.endswith(".mmcif"):
              tmpPdb = filePath + ".tmp.pdb"
              cifToPdb(filePath, tmpPdb)
              parsePath = tmpPdb
          else:
              parsePath = filePath

          structObj = parser.get_structure(f"struct_{i}", parsePath)

          chain_seqs = []

          for model in structObj:
              for chain in model:

                  residues = []
                  for res in chain:
                      if res.id[0] != " ":
                          continue
                      residues.append(res.get_resname())

                  try:
                      seq = "".join(seq1(r) for r in residues)
                      if len(seq) > 0:
                          chain_seqs.append(seq)
                  except Exception:
                      continue

          key = os.path.basename(struct.getFileName()).split('.')[0]

          seqsDic[key] = "".join(chain_seqs)

          if parsePath != filePath:
              os.remove(parsePath)

      return seqsDic
