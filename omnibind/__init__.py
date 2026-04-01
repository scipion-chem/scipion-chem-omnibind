# **************************************************************************
# *
# * Authors:  Blanca Pueche (blanca.pueche@cnb.csic.es)
# *
# * Biocomputing Unit, CNB-CSIC
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

import shutil

from scipion.install.funcs import InstallHelper

from pwchem import Plugin as pwchemPlugin
from .constants import *

_references = ['']


class Plugin(pwchemPlugin):
    @classmethod
    def defineBinaries(cls, env):
        cls.addOmniBindPackage(env)

    @classmethod
    def _defineVariables(cls):
        """ Return and write a variable in the config file.
        """
        cls._defineEmVar(OMNIBIND_DIC['home'], cls.getEnvName(OMNIBIND_DIC))

    @classmethod
    def addOmniBindPackage(cls, env, default=True):
        installer = InstallHelper(
            OMNIBIND_DIC['name'],
            packageHome=cls.getVar(OMNIBIND_DIC['home']),
            packageVersion=OMNIBIND_DIC['version']
        )

        installer.getCondaEnvCommand(
            OMNIBIND_DIC['name'],
            binaryVersion=OMNIBIND_DIC['version'],
            pythonVersion='3.11'
        ).addCommand(
            "git clone https://github.com/Shimizu-team/OmniBind.git",
            f"{OMNIBIND_DIC['name']}_cloned"
        ).addCommand(
            f"{cls.getEnvActivationCommand(OMNIBIND_DIC)} && "
            "pip install torch torchvision torchaudio"
        ).addCommand(
            f"{cls.getEnvActivationCommand(OMNIBIND_DIC)} && "
            "conda install -c conda-forge rdkit -y"
        ).addCommand(
            f"{cls.getEnvActivationCommand(OMNIBIND_DIC)} && "
            "cd OmniBind && pip install -r requirements.txt && pip install -e ."
        )

        installer.addCommand(
            "cd OmniBind/checkpoints && "
            "wget https://zenodo.org/records/19326040/files/checkpoints.zip -O checkpoints.zip",
            "omnibind_checkpoint_zip_downloaded"
        ).addCommand(
            "cd OmniBind/checkpoints && "
            "unzip -o checkpoints.zip && "
            "rm -rf __MACOSX checkpoints.zip && "
            "find checkpoints -name '.DS_Store' -delete && "
            "find checkpoints -name '._*' -delete && "
            "mv checkpoints/application.pth ./application.pth && "
            "rm -rf checkpoints",
            "omnibind_application_model_ready"
        )

        installer.addPackage(
            env,
            dependencies=['git', 'pip', 'conda'],
            default=default
        )

