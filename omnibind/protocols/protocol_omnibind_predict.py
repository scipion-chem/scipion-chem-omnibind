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
import shutil
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from pwchem.objects import SetOfSmallMolecules, SmallMolecule
from pwem.objects import AtomStruct, SetOfAtomStructs

from pwem.protocols import EMProtocol
from pyworkflow.protocol import params

from pwem.convert import cifToPdb
from pyworkflow.object import String, Float
from pwchem import Plugin as pwchemPlugin
from pwchem.constants import OPENBABEL_DIC

from .. import Plugin as omnibindPlugin
from ..constants import OMNIBIND_DIC

class ProtOmniBindPrediction(EMProtocol):
  """Run a prediction using a OmniBind trained model over a set of proteins and ligands"""
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

    iGroup = form.addGroup('Input') #todo choose whether to input one AtomStruct or a set and then the output will depend on the chosen input
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
    if self.input.get() == 0:
        self._insertFunctionStep(self.createOutputStepSingle)
    else:
        self._insertFunctionStep(self.createOutputStepSet)


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

  def createOutputStepSingle(self):
      resultsFile = os.path.abspath(os.path.join(self.getPath(), "results.csv"))

      smisDic = self.getInputSMIs()

      results_map = {}
      with open(resultsFile, 'r') as f:
          reader = csv.DictReader(f)
          for row in reader:
              key = (row['protein'], row['smiles'].strip())
              results_map[key] = row

      outputMols = SetOfSmallMolecules().create(outputPath=self._getPath())

      protID = os.path.basename(self.inputStructure.get().getFileName()).split('.')[0]

      for mol in self.inputSmallMols.get():
          newMol = SmallMolecule()
          newMol.copy(mol)

          newMol.setProteinFile(self.inputStructure.get().getFileName())
          molName = os.path.basename(mol.getFileName()).split('.')[0]

          molSmi = smisDic.get(molName)

          if molSmi:
              predKey = (protID, molSmi.strip())
              if predKey in results_map:
                  res = results_map[predKey]
                  newMol.pKi = Float()
                  newMol.setAttributeValue('pKi', float(res['pKi']))
                  newMol.pKd = Float()
                  newMol.setAttributeValue('pKd', float(res['pKd']))
                  newMol.pIC50 = Float()
                  newMol.setAttributeValue('pIC50', float(res['pIC50']))
                  newMol.pEC50 = Float()
                  newMol.setAttributeValue('pEC50', float(res['pEC50']))

          outputMols.append(newMol)

      self._defineOutputs(outputSmallMols=outputMols)

  def createOutputStepSet(self):
      resultsFile = os.path.abspath(os.path.join(self.getPath(), "results.csv"))

      if not os.path.exists(resultsFile):
          self.getLogger().info("OmniBind results file not found!")
          return

      smisDic = self.getInputSMIs()

      results_map = {}
      with open(resultsFile, 'r') as f:
          reader = csv.DictReader(f)
          for row in reader:
              key = (row['protein'], row['smiles'].strip())
              results_map[key] = row

      outputMols = SetOfSmallMolecules().create(outputPath=self._getPath())

      for struct in self.inputStructures.get():
          protPath = struct.getFileName()
          protID = os.path.basename(protPath).split('.')[0]

          for mol in self.inputSmallMols.get():
              newMol = SmallMolecule()
              newMol.copy(mol, copyId=False)

              newMol.setProteinFile(protPath)

              molName = os.path.basename(mol.getFileName()).split('.')[0]
              molSmi = smisDic.get(molName)

              if molSmi:
                  predKey = (protID, molSmi.strip())
                  if predKey in results_map:
                      res = results_map[predKey]

                      newMol.pKi = Float()
                      newMol.setAttributeValue('pKi', float(res['pKi']))
                      newMol.pKd = Float()
                      newMol.setAttributeValue('pKd', float(res['pKd']))
                      newMol.pIC50 = Float()
                      newMol.setAttributeValue('pIC50', float(res['pIC50']))
                      newMol.pEC50 = Float()
                      newMol.setAttributeValue('pEC50', float(res['pEC50']))

              outputMols.append(newMol)

      self._defineOutputs(outputSmallMols=outputMols)

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
