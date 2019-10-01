import numpy as np


class Filterbank(np.ndarray):
    """
    Generic filterbank class.
    A Filterbank is a simple numpy array enhanced with several additional
    attributes, e.g. number of bands.
    A Filterbank has a shape of (num_bins, num_bands) and can be used to
    filter a spectrogram of shape (num_frames, num_bins) to (num_frames,
    num_bands).
    Parameters
    ----------
    data : numpy array, shape (num_bins, num_bands)
        Data of the filterbank .
    bin_frequencies : numpy array, shape (num_bins, )
        Frequencies of the bins [Hz].
    Notes
    -----
    The length of `bin_frequencies` must be equal to the first dimension
    of the given `data` array.
    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, data, bin_frequencies):
        # this method is for documentation purposes only
        pass

    def __new__(cls, data, bin_frequencies):
        # input is an numpy ndarray instance
        if isinstance(data, np.ndarray) and data.ndim == 2:
            # cast as Filterbank
            obj = np.asarray(data, dtype=FILTER_DTYPE).view(cls)
        else:
            raise TypeError('wrong input data for Filterbank, must be a 2D '
                            'np.ndarray')
        # set bin frequencies
        if len(bin_frequencies) != obj.shape[0]:
            raise ValueError('`bin_frequencies` must have the same length as '
                             'the first dimension of `data`.')
        obj.bin_frequencies = np.asarray(bin_frequencies, dtype=np.float)
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.bin_frequencies = getattr(obj, 'bin_frequencies', None)

    @classmethod
    def _put_filter(cls, filt, band):
        """
        Puts a filter in the band, internal helper function.
        Parameters
        ----------
        filt : :class:`Filter` instance
            Filter to be put into the band.
        band : numpy array
            Band in which the filter should be put.
        Notes
        -----
        The `band` must be an existing numpy array where the filter `filt` is
        put in, given the position of the filter. Out of range filters are
        truncated. If there are non-zero values in the filter band at the
        respective positions, the maximum value of the `band` and the filter
        `filt` is used.
        """
        if not isinstance(filt, Filter):
            raise ValueError('unable to determine start position of Filter')
        # determine start and stop positions
        start = filt.start
        stop = start + len(filt)
        # truncate the filter if it starts before the 0th band bin
        if start < 0:
            filt = filt[-start:]
            start = 0
        # truncate the filter if it ends after the last band bin
        if stop > len(band):
            filt = filt[:-(stop - len(band))]
            stop = len(band)
        # put the filter in place
        filter_position = band[start:stop]
        # TODO: if needed, allow other handling (like summing values)
        np.maximum(filt, filter_position, out=filter_position)

    @classmethod
    def from_filters(cls, filters, bin_frequencies):
        """
        Create a filterbank with possibly multiple filters per band.
        Parameters
        ----------
        filters : list (of lists) of Filters
            List of Filters (per band); if multiple filters per band are
            desired, they should be also contained in a list, resulting in a
            list of lists of Filters.
        bin_frequencies : numpy array
            Frequencies of the bins (needed to determine the expected size of
            the filterbank).
        Returns
        -------
        filterbank : :class:`Filterbank` instance
            Filterbank with respective filter elements.
        """
        # create filterbank
        fb = np.zeros((len(bin_frequencies), len(filters)))
        # iterate over all filters
        for band_id, band_filter in enumerate(filters):
            # get the band's corresponding slice of the filterbank
            band = fb[:, band_id]
            # if there's a list of filters for the current band, put them all
            # into this band
            if isinstance(band_filter, list):
                for filt in band_filter:
                    cls._put_filter(filt, band)
            # otherwise put this filter into that band
            else:
                cls._put_filter(band_filter, band)
        # create Filterbank and cast as class where this method was called from
        return Filterbank.__new__(cls, fb, bin_frequencies)

    @property
    def num_bins(self):
        """Number of bins."""
        return self.shape[0]

    @property
    def num_bands(self):
        """Number of bands."""
        return self.shape[1]

    @property
    def corner_frequencies(self):
        """Corner frequencies of the filter bands."""
        freqs = []
        for band in range(self.num_bands):
            # get the non-zero bins per band
            bins = np.nonzero(self[:, band])[0]
            # append the lowest and highest bin
            freqs.append([np.min(bins), np.max(bins)])
        # map to frequencies
        return bins2frequencies(freqs, self.bin_frequencies)

    @property
    def center_frequencies(self):
        """Center frequencies of the filter bands."""
        freqs = []
        for band in range(self.num_bands):
            # get the non-zero bins per band
            bins = np.nonzero(self[:, band])[0]
            min_bin = np.min(bins)
            max_bin = np.max(bins)
            # if we have a uniform filter, use the center bin
            if self[min_bin, band] == self[max_bin, band]:
                center = int(min_bin + (max_bin - min_bin) / 2.)
            # if we have a filter with a peak, use the peak bin
            else:
                center = min_bin + np.argmax(self[min_bin: max_bin, band])
            freqs.append(center)
        # map to frequencies
        return bins2frequencies(freqs, self.bin_frequencies)

    @property
    def fmin(self):
        """Minimum frequency of the filterbank."""
        return self.bin_frequencies[np.nonzero(self)[0][0]]

    @property
    def fmax(self):
        """Maximum frequency of the filterbank."""
        return self.bin_frequencies[np.nonzero(self)[0][-1]]
