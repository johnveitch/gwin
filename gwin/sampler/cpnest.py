# Copyright (C) 2018  John Veitch
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""
This modules provides classes and functions for using the emcee sampler
packages for parameter estimation.
"""

from __future__ import absolute_import

import numpy
from pycbc.io import FieldArray
from pycbc.filter import autocorrelation

from .base import BaseSampler

from abc import ABCMeta, abstractmethod, abstractproperty

import cpnest

class BaseNestedSampler(object):
    """This class provides methods common to Nested Samplers.

    It is not a sampler class itself. Sampler classes can inherit from this
    along with ``BaseSampler``.
    """
    __metaclass__ = ABCMeta
    _pos = None
    
    @property
    def niterations(self):
        """Get the current number of iterations."""
        pass
    
    @property
    def pos(self):
        """
        Return samples from the posterior
        """
        pass
    
    @property
    def log_evidence(self):
        """
        Return the log evidence
        """
        pass

class CPNestModelWrapper(cpnest.Model):
    """
    Class that wraps a gwin.BaseModel to interface with CPNest
    """
    def __init__(self, gwin_model):
        self._gwin = gwin_model
        self.names = gwin_model.sampling_params
        self.bounds = None # Need to implement a bounding box in gwin's model or prior
        
    def log_prior(self, params):
        return self._gwin.prior(params)
    
    def log_likelihood(self, params):
        return self._gwin.loglikelihood(params)
        

class CPNestSampler(BaseNestedSampler,BaseSampler):
    """
    This class provides an interface to CPNest.
    """
    def __init__(self, model, nlive, nthreads, **kwargs):
        super(self, BaseSampler).__init__(model)
        
        self._cpnestmodel = CPNestModelWrapper(self.model)
        self._cpnest = CPNest(self._cpnestmodel, Nlive = nlive)
        
    def from_config(cls, cp, model, pool=None, model_call=None, **kwargs):
        pass
    
    @property
    def raw_samples(self):
        return self._cpnest.nested_samples
    
    def run(self):
        self._cpnest.run()
    
