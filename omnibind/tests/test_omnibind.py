# **************************************************************************
# *
# * Authors:     Daniel Del Hoyo (ddelhoyo@cnb.csic.es)
# *
# * Unidad de Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307 USA
# *
# * All comments concerning this program package may be sent to the
# * e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
from pwem.protocols import ProtImportPdb
from pyworkflow.tests import setupTestProject, DataSet

# Scipion chem imports
from pwchem.protocols import ProtChemImportSmallMolecules
from pwchem.utils import assertHandle
from pyworkflow.tests import BaseTest

from ..protocols import ProtOmniBindPrediction

class TestOmniBindPrediction(BaseTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.ds = DataSet.getDataSet('model_building_tutorial')
        cls.dsLig = DataSet.getDataSet("smallMolecules")
        setupTestProject(cls)

        cls._runImportSmallMols()
        cls._runImportPDB()
        cls._waitOutput(cls.protImportSmallMols, 'outputSmallMolecules', sleepTime=5)


    @classmethod
    def _runImportPDB(cls):
        protImportPDB = cls.newProtocol(
            ProtImportPdb,
            inputPdbData=1, pdbFile=cls.ds.getFile('PDBx_mmCIF/1aoi.cif'))
        cls.launchProtocol(protImportPDB)
        cls.protImportPDB = protImportPDB

    @classmethod
    def _runImportSmallMols(cls):
        cls.protImportSmallMols = cls.newProtocol(
            ProtChemImportSmallMolecules,
            filesPath=cls.dsLig.getFile('mol2'))
        cls.proj.launchProtocol(cls.protImportSmallMols, wait=False)

    def _runOmniBindPrediction(self):
        protOmniBind = self.newProtocol(ProtOmniBindPrediction)

        protOmniBind.inputStructure.set(self.protImportPDB)
        protOmniBind.inputStructure.setExtended('outputPdb')
        protOmniBind.inputSmallMols.set(self.protImportSmallMols)
        protOmniBind.inputSmallMols.setExtended('outputSmallMolecules')

        self.proj.launchProtocol(protOmniBind, wait=False)
        return protOmniBind

    def test(self):
        protOmniBind = self._runOmniBindPrediction()
        self._waitOutput(protOmniBind, 'outputAtomStructs', sleepTime=10)
        assertHandle(self.assertIsNotNone, getattr(protOmniBind, 'outputAtomStructs', None))
