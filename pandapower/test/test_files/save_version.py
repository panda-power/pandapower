# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapower.networks as nw
import pandapower as pp
import os

net = nw.example_multivoltage()
pp.runpp(net)
pp.to_json(net, os.path.join("old_versions", "example_%s.json"%pp.__version__))