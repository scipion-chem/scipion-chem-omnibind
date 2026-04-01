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

from pyworkflow.tests import setupTestProject, DataSet

# Scipion chem imports
from pwchem.protocols import ProtChemImportSmallMolecules
from pwchem.tests import TestImportSequences
from pwchem.utils import assertHandle

from ..protocols import ProtConPLexPrediction

class TestConPLexPrediction(TestImportSequences):
	@classmethod
	def setUpClass(cls):
		super().setUpClass()
		cls.ds = DataSet.getDataSet('model_building_tutorial')
		cls.dsLig = DataSet.getDataSet("smallMolecules")
		setupTestProject(cls)

		cls._runImportSmallMols()
		cls._runImportSeqs()
		cls._waitOutput(cls.protImportSmallMols, 'outputSmallMolecules', sleepTime=5)
		cls._waitOutput(cls.protImportSeqs, 'outputSequences', sleepTime=5)

	@classmethod
	def _runImportSmallMols(cls):
		cls.protImportSmallMols = cls.newProtocol(
			ProtChemImportSmallMolecules,
			filesPath=cls.dsLig.getFile('mol2'))
		cls.proj.launchProtocol(cls.protImportSmallMols, wait=False)

	def _runConPLexPrediction(self):
		protConPLex = self.newProtocol(ProtConPLexPrediction)

		protConPLex.inputSequences.set(self.protImportSeqs)
		protConPLex.inputSequences.setExtended('outputSequences')
		protConPLex.inputSmallMols.set(self.protImportSmallMols)
		protConPLex.inputSmallMols.setExtended('outputSmallMolecules')

		self.proj.launchProtocol(protConPLex, wait=False)
		return protConPLex

	def test(self):
		protConPLex = self._runConPLexPrediction()
		self._waitOutput(protConPLex, 'outputSequences', sleepTime=10)
		assertHandle(self.assertIsNotNone, getattr(protConPLex, 'outputSequences', None))
