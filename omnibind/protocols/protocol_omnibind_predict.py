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
import csv
import json
import os
import pickle
import shutil
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from pwchem.objects import SetOfSmallMolecules, SmallMolecule, SmallMoleculesLibrary
from pwem.objects import SetOfAtomStructs

from pwem.protocols import EMProtocol
from pyworkflow.protocol import params

from pwem.convert import cifToPdb
from pyworkflow.object import String, Float, Pointer
from pwchem import Plugin as pwchemPlugin
from pwchem.constants import OPENBABEL_DIC

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
    #if self.input.get() == 0:
    #    self._insertFunctionStep(self.createOutputStepSingle)
    #else:
    #    self._insertFunctionStep(self.createOutputStepSet)
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
          dst = os.path.join(structDir, os.path.basename(src))
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

  def createOutputStep(self):
      resultsFile = self.getPath("results.csv")
      smisDic = self.getInputSMIs()
      smiToName = {v.strip(): k for k, v in smisDic.items()}

      intDic = {}
      with open(resultsFile, 'r') as f:
          reader = csv.DictReader(f, delimiter=',')
          for row in reader:
              protID = row['protein'].strip()
              smi = row['smiles'].strip()
              molName = smiToName.get(smi)

              if not molName:
                  continue

              if protID not in intDic:
                  intDic[protID] = {}
              intDic[protID][molName] = {
                  "OmniBind_pKi": float(row['pKi']),
                  "OmniBind_pKd": float(row['pKd']),
                  "OmniBind_pIC50": float(row['pIC50']),
                  "OmniBind_pEC50": float(row['pEC50'])
              }

      inStructs = self._getInpStructs()
      outStructs = SetOfAtomStructs().create(outputPath=self._getPath())

      outputFile = self.writeInteractScoresDic(intDic)
      outStructs._interactScoresFile = String(outputFile)

      data = {}
      for struct in inStructs:
          protID = os.path.basename(struct.getFileName()).split('.')[0]

          outStruct = struct.clone()
          #outStruct._interactScoresFile = String(outputFile)
          outStructs.append(outStruct)

          if protID in intDic:
              data[protID] = intDic[protID]


      outMols = self.inputLibrary.get() if self.useLibrary.get() else self.inputSmallMols.get()
      #molsListFile = self.setInteractMols(mols=outMols, structs=outStructs)
      #outStructs._interactMols = String(molsListFile)

      allMolsSet = SetOfSmallMolecules().create(outputPath=self._getPath())
      addedMolsNames = set()
      inputObj = self.inputStructure.get() if self.input.get() == 0 else self.inputStructures.get()
      if hasattr(inputObj,'_interactMols'):
          prevMols = inputObj._interactMols.get()
          if prevMols:
              for m in prevMols:
                  allMolsSet.append(m)
                  addedMolsNames.add(m.getMolName())

      currentMols = self.inputLibrary.get() if self.useLibrary.get() else self.inputSmallMols.get()
      for m in currentMols:
          if m.getMolName() not in addedMolsNames:
              allMolsSet.append(m)
              addedMolsNames.add(m.getMolName())


      outStructs._interactMols = Pointer(allMolsSet)

      self._defineOutputs(outputAtomStructs=outStructs)

      if len(inStructs) == 1:
          protID = os.path.basename(inStructs[0].getFileName()).split('.')[0]
          scoreDic = intDic.get(protID, {})

          if self.useLibrary.get():
              inLib = self.inputLibrary.get()
              mapDic = inLib.getLibraryMap(inverted=True, fullLine=True)
              oLibFile = self._getPath('outputLibrary.smi')

              with open(oLibFile, 'w') as f:
                  for molName, scores in scoreDic.items():
                      f.write(f'{mapDic[molName]}\t{scores["OmniBind_pKi"]}\n')

              outputLib = inLib.clone()
              outputLib.setFileName(oLibFile)
              outputLib.setHeaders(inLib.getHeaders() + ['OmniBind_pKi'])
              self._defineOutputs(outputLibrary=outputLib)

          else:
              inSet = self.inputSmallMols.get()
              outputSet = inSet.createCopy(self._getPath(), copyInfo=True)

              for mol in inSet:
                  nMol = mol.clone()
                  molName = nMol.getMolName()
                  if molName in scoreDic:
                      s = scoreDic[molName]
                      setattr(nMol, 'OmniBind_pKi', Float(s['OmniBind_pKi']))
                      setattr(nMol, 'OmniBind_pKd', Float(s['OmniBind_pKd']))
                      setattr(nMol, 'OmniBind_pIC50', Float(s['OmniBind_pIC50']))
                      setattr(nMol, 'OmniBind_pEC50', Float(s['OmniBind_pEC50']))
                      outputSet.append(nMol)

              outputSet.updateMolClass()
              self._defineOutputs(outputSmallMolecules=outputSet)


  ############## UTILS ########################
  def setInteractMols(self, mols, structs):
      molsListFile = os.path.join(self._getExtraPath(), 'interacting_molecules.txt')
      allPaths = set()

      inputObj = self.inputStructure.get() if self.input.get() == 0 else self.inputStructures.get()

      if hasattr(inputObj, '_interactMols'):
          prevFile = inputObj.getAttributeValue('_interactMols')
          if prevFile and os.path.exists(str(prevFile)):
              print(f"--- DEBUG: Leyendo rastro previo de {prevFile} ---")
              with open(str(prevFile), 'r') as f:
                  allPaths.update(line.strip() for line in f if line.strip())

      for mol in mols:
          molPath = mol.getFileName()
          if molPath:
              allPaths.add(os.path.abspath(molPath))

      with open(molsListFile, 'w') as f:
          for path in sorted(allPaths):
              f.write(f"{path}\n")

      structs._interactMols = String(molsListFile)

      return molsListFile

  def writeInteractScoresDic(self, intDic, outFile=None):
      """
      Generates/Updates a JSON file. If a previous scores file exists,
      it merges the new data into it.
      """
      if not outFile:
          outFile = os.path.join(self._getExtraPath(), 'scoresFile.json')

      finalData = {}

      inStructs = self._getInpStructs()
      if hasattr(inStructs, '_interactScoresFile'):
          prevFile = getattr(inStructs, '_interactScoresFile')
      else:
          prevFile = None

      if prevFile:
          try:
              with open(str(prevFile), 'r') as f:
                  finalData = json.load(f)
          except Exception:
              finalData = {}

      for protID, newMols in intDic.items():
          if protID not in finalData:
              finalData[protID] = {}

          for molName, newScores in newMols.items():
              if molName in finalData[protID]:
                  finalData[protID][molName].update(newScores)
              else:
                  finalData[protID][molName] = newScores

      with open(outFile, 'w') as f:
          json.dump(finalData, f, indent=4)

      return outFile

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
