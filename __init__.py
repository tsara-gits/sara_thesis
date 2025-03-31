"""
OpenCGChromatin
===============

A package for coarse-grained simulations using the Collepardo lab's GPU-accelerated chemically specific chromatin model, implemented in OpenMM.

Modules:
- biomolecules: Defines classes for various biomolecules.
- coordinate_building: Provides functions for building/reading initial configurations.
- system_building: Contains functions for building OpenMM System objects.
- constants: Useful constants and functions for data loading

Author:
- Kieran Russell kor20@cam.ac.uk
- Collepardo Lab, University of Cambridge

Version: 0.1.0
"""

PACKAGE_NAME = 'OpenCGChromatin'
__version__ = "0.1.0"

import json
import os
import numpy as np
import openmm as mm
from openmm import app
from openmm import unit

from .biomolecules import IDP, MDP, DNA, NucleosomeCore, NucleosomeArray
from .constants import NUCLEOSOME_DATA, TETRAMER_DATA, PLATFORM, PROPERTIES
from .system_building import get_system, get_minimized_system
