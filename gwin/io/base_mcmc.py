# Copyright (C) 2016 Christopher M. Biwer, Collin Capano
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# self.option) any later version.
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
"""Provides I/O that is specific to MCMC samplers.
"""

import os
import sys
import logging
from abc import ABCMeta

import numpy

import h5py

from pycbc import DYN_RANGE_FAC
from pycbc.io import FieldArray
from pycbc.types import FrequencySeries
from pycbc.waveform import parameters as wfparams

from .hdf import InferenceFile

class EnsembleMCMCIO(obect):
    """Abstract base class that provides some IO functions for ensemble MCMCs.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def read_acls(self):
        """Should return all of the individual chains' acls.
        """
        pass

    def write_mcmc_metadata(self, sampler):
        """Writes metadata unique to an ensemble MCMC.

        Parameters
        ----------
        sampler : gwin.sampler
            An instance of a gwin sampler.
        """
        self.attrs["niterations"] = sampler.niterations
        self.attrs["nwalkers"] = sampler.nwalkers

    def write_samples(self, parameters, samples,
                      start_iteration=None, max_iterations=None):
        """Writes samples to the given file.

        Results are written to:

            ``fp[samples_group/{vararg}]``,

        where ``{vararg}`` is the name of a model params. The samples are
        written as an ``nwalkers x niterations`` array.

        Parameters
        -----------
        parameters : list
            The parameters to write to the file.
        samples : dict
            The samples to write. Each array in the dictionary should have
            shape nwalkers x niterations.
        start_iteration : int, optional
            Write results to the file's datasets starting at the given
            iteration. Default is to append after the last iteration in the
            file.
        max_iterations : int, optional
            Set the maximum size that the arrays in the hdf file may be resized
            to. Only applies if the samples have not previously been written
            to file. The default (None) is to use the maximum size allowed by
            h5py.
        """
        nwalkers, niterations = samples.values()[0].shape
        assert(all(p.shape == (nwalkers, niterations)
                   for p in samples.values()),
               "all samples must have the same shape")
        if max_iterations is not None and max_iterations < niterations:
            raise IndexError("The provided max size is less than the "
                             "number of iterations")
        group = self.samples_group + '/{name}'
        # loop over number of dimensions
        for param in parameters:
            dataset_name = group.format(name=param)
            istart = start_iteration
            try:
                fp_niterations = fp[dataset_name].shape[-1]
                if istart is None:
                    istart = fp_niterations
                istop = istart + niterations
                if istop > fp_niterations:
                    # resize the dataset
                    fp[dataset_name].resize(istop, axis=1)
            except KeyError:
                # dataset doesn't exist yet
                if istart is not None and istart != 0:
                    raise ValueError("non-zero start_iteration provided, "
                                     "but dataset doesn't exist yet")
                istart = 0
                istop = istart + niterations
                fp.create_dataset(dataset_name, (nwalkers, istop),
                                  maxshape=(nwalkers, max_iterations),
                                  dtype=float, fletcher32=True)
            fp[dataset_name][:, istart:istop] = samples[param]

    def _read_samples_data(self, fields,
                           thin_start=None, thin_interval=None, thin_end=None,
                           iteration=None, walkers=None, flatten=True):
        """Base function for reading samples.

        Parameters
        -----------
        fields : list
            The list of field names to retrieve. Must be names of datasets in
            the ``samples_group``.

        Returns
        -------
        dict
            A dictionary of field name -> numpy array pairs.
        """
        # walkers to load
        if walkers is not None:
            widx = numpy.zeros(fp.nwalkers, dtype=bool)
            widx[walkers] = True
        else:
            widx = slice(0, None)
        # get the slice to use
        if iteration is not None:
            get_index = iteration
        else:
            if thin_end is None:
                # use the number of current iterations
                thin_end = fp.niterations
            get_index = fp.get_slice(thin_start=thin_start, thin_end=thin_end,
                                     thin_interval=thin_interval)
        # load
        group = self.samples_group + '/{name}'
        arrays = {}
        for name in fields:
            arr = fp[group.format(name=name)][widx, get_index]
            if flatten:
                arr = arr.flatten()
            arrays[name] = arr
        return arrays

    def write_resume_point(self):
        """Keeps a list of the number of iterations that were in a file when a
        run was resumed from a checkpoint."""
        try:
            resume_pts = self.attrs["resume_points"].tolist()
        except KeyError:
            resume_pts = []
        try:
            niterations = self.niterations
        except KeyError:
            niterations = 0
        resume_pts.append(niterations)
        self.attrs["resume_points"] = resume_pts

