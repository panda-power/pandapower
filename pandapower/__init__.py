__version__ = "2.2.1"

import os
pp_dir = os.path.dirname(os.path.realpath(__file__))

from pandapower.auxiliary import *
from pandapower.convert_format import *
from pandapower.create import *
from pandapower.diagnostic import *
from pandapower.file_io import *
from pandapower.run import *
from pandapower.runpm import *
from pandapower.std_types import *
from pandapower.toolbox import *
from pandapower.powerflow import *
from pandapower.opf import *
from pandapower.optimal_powerflow import OPFNotConverged

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
