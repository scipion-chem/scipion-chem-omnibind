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

from pwem.protocols import EMProtocol
from pyworkflow.protocol import params

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
    iGroup.addParam('inputSequences', params.PointerParam, pointerClass="SetOfSequences",
                    label='Input protein sequences: ',
                    help="Set of protein sequences to perform the screening on")

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

  def predictStep(self):
    smisDic = self.getInputSMIs()
    protSeqsDic = self.getInputSeqs()

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
        "model_type": "aa3di_gmf", #todo issue opened to try to fix this
        "compounds_csv": compoundsFile,
        "config": os.path.join(omnibindPlugin.getVar(OMNIBIND_DIC['home']), "OmniBind/configs/default.yaml"),
        "output": os.path.abspath(os.path.join(self.getPath(), "results.csv")),
        "sequences": protSeqsDic,
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
      inSeqs = self.inputSequences.get()

      if not self.useLibrary.get():
          inMols = self.inputSmallMols.get()
      else:
          inMols = self.inputLibrary.get()

      resultsFile = self.getPath('results.csv')
      df = pd.read_csv(resultsFile)

      compoundsFile = os.path.abspath(self._getExtraPath('compounds.csv'))
      comp_df = pd.read_csv(compoundsFile)

      smiles_to_id = {}
      for _, row in comp_df.iterrows():
          smiles = row.get("smiles")
          mol_id = row.get("id")
          if smiles and mol_id:
              smiles_to_id[smiles] = mol_id

      intDic = {}

      for _, row in df.iterrows():

          seqName = row.get("protein")
          smiles = row.get("smiles")

          if not seqName or not smiles:
              continue

          molName = smiles_to_id.get(smiles)
          if molName is None:
              continue

          intDic.setdefault(seqName, {})
          intDic[seqName].setdefault(molName, {})

          intDic[seqName][molName] = {
              "pKi": float(row["pKi"]),
              "pKd": float(row["pKd"]),
              "pIC50": float(row["pIC50"]),
              "pEC50": float(row["pEC50"])
          }

      try:
          outputFile = inSeqs.getInteractScoresFile()
      except Exception:
          outputFile = None

      if outputFile and os.path.exists(outputFile):
          localFile = self._getExtraPath("scoresFile.json")
          shutil.copy(outputFile, localFile)
          outputFile = localFile

          with open(outputFile, "r") as f:
              data = json.load(f)

          existing = {
              e.get("sequence"): e
              for e in data.get("entries", [])
              if e.get("sequence")
          }
          newEntries = []

          for seq in inSeqs:
              seqName = seq.getSeqName()

              if seqName not in existing:
                  existing[seqName] = {
                      "sequence": seqName,
                      "molecules": {}
                  }

              seqMolScores = intDic.get(seqName, {})

              for molName, scores in seqMolScores.items():
                  molEntry = existing[seqName]["molecules"].setdefault(molName, {})

                  molEntry["score_OmniBind"] = {
                      "pKi": scores["pKi"],
                      "pKd": scores["pKd"],
                      "pIC50": scores["pIC50"],
                      "pEC50": scores["pEC50"]
                  }

              newEntries.append(existing[seqName])

          inSeqs.setInteractScoresDic(newEntries, data, outputFile)

      else:
          data = {"entries": []}

          for seqName, mols in intDic.items():

              entry = {
                  "sequence": seqName,
                  "molecules": {}
              }
              for molName, scores in mols.items():
                  entry["molecules"][molName] = {
                      "score_OmniBind": {
                          "pKi": scores["pKi"],
                          "pKd": scores["pKd"],
                          "pIC50": scores["pIC50"],
                          "pEC50": scores["pEC50"]
                      }
                  }

              data["entries"].append(entry)
          outputFile = self._getExtraPath("scoresFile.json")

          with open(outputFile, "w") as f:
              json.dump(data, f, indent=2)

      outSeqs = SetOfSequencesChem().create(outputPath=self._getPath())

      for seq in inSeqs:
          outSeq = SequenceChem()
          outSeq.copy(seq)

          outSeq.setInteractScoresFile(outputFile)
          outSeqs.append(outSeq)

      outSeqs.setInteractMols(mols=inMols)
      outSeqs.setScoreTypes(scores=["OmniBind"])

      for outSeq in outSeqs:
          outSeq.setInteractScoresFile(str(outputFile))
      outSeqs.setInteractScoresFile(outputFile)

      self._defineOutputs(outputSequences=outSeqs)

      if len(inSeqs) == 1:

          inSeq = inSeqs.getFirstItem()
          seqName = inSeq.getSeqName()

          scoreDic = intDic.get(seqName, {})

          if self.useLibrary.get():

              mapDic = self.inputLibrary.get().getLibraryMap(inverted=True)

              oLibFile = self._getPath('outputLibrary.smi')

              with open(oLibFile, 'w') as f:
                  for molName, scores in scoreDic.items():
                      f.write(
                          f'{mapDic.get(molName, molName)}\t'
                          f'{molName}\t'
                          f'{scores["pKi"]}\n'
                      )

              outputLib = SmallMoleculesLibrary(
                  libraryFilename=oLibFile,
                  origin='GCR'
              )

              self._defineOutputs(outputLibrary=outputLib)

          else:

              inSet = self.inputSmallMols.get()
              outputSet = inSet.createCopy(self._getPath(), copyInfo=True)

              for mol in outputSet:

                  molName = mol.getMolName()

                  if molName in scoreDic:
                      scores = scoreDic[molName]

                      setattr(mol, "_omnibind_pKi", params.Float(scores["pKi"]))
                      setattr(mol, "_omnibind_pKd", params.Float(scores["pKd"]))
                      setattr(mol, "_omnibind_pIC50", params.Float(scores["pIC50"]))
                      setattr(mol, "_omnibind_pEC50", params.Float(scores["pEC50"]))

              outputSet.updateMolClass()
              self._defineOutputs(outputSmallMolecules=outputSet)


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

  def getInputSeqs(self):
      seqsDic = {}
      for i, seq in enumerate(self.inputSequences.get()):
          key = seq.getId()
          if not key:
              key = f"seq_{i}"
          seqsDic[key] = seq.getSequence()
      return seqsDic

  def getInteractionsFile(self):
    return self.getPath('results.tsv')

  def parseInteractionsFile(self, iFile):
    '''Return a dictionary of the form {seqName: {molName: score}}'''
    intDic, molNames = {}, set([])
    with open(iFile) as f:
      for line in f:
        molName, seqName, score = line.strip().split('\t')
        molNames.add(molName)
        if seqName in intDic:
          intDic[seqName][molName] = score
        else:
          intDic[seqName] = {molName: score}

    seqNames = list(intDic.keys())
    molNames = list(molNames)
    seqNames.sort(), molNames.sort()

    return intDic, seqNames, molNames

