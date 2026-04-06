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

import json
import os
import shutil
import pandas as pd

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from pwem.objects import AtomStruct, SetOfAtomStructs

from pwem.protocols import EMProtocol
from pyworkflow.protocol import params

from pwem.convert import cifToPdb
from pyworkflow.object import String
from pwchem import Plugin as pwchemPlugin
from pwchem.constants import OPENBABEL_DIC
from pwchem.objects import SequenceChem, SetOfSequencesChem, SmallMoleculesLibrary

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

    iGroup = form.addGroup('Input')
    #iGroup.addParam('inputSequences', params.PointerParam, pointerClass="SetOfSequences",
    #                label='Input protein sequences: ',
    #                help="Set of protein sequences to perform the screening on")
    iGroup.addParam('inputStructures', params.PointerParam, pointerClass="SetOfAtomStructs",
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
      structSet = self.inputStructures.get()

      structDir = os.path.abspath(self._getExtraPath('structures'))
      os.makedirs(structDir, exist_ok=True)

      for struct in structSet:
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
      inpStructs = self.inputStructures.get()
      resultsFile = self._getPath('results.csv')
      outStructs = SetOfAtomStructs().create(outputPath=self._getPath())

      for struct in inpStructs:
          newStruct = AtomStruct()
          newStruct.copy(struct)

          newStruct.OmniBind_file = String()
          newStruct.setAttributeValue('OmniBind_file', str(resultsFile))
          outStructs.append(newStruct)

      self._defineOutputs(outputAtomStructs=outStructs)

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

  def getInputSeqsFromStructures(self):
      seqsDic = {}

      parser = PDBParser(QUIET=True)

      for i, struct in enumerate(self.inputStructures.get()):
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
