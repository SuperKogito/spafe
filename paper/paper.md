---
title: 'Spafe: Simplified python audio features extraction'
tags:
  - Python
  - Signal processing
  - time-frequency analysis
  - audio features extraction
authors:
  - name: Ayoub Malek
    orcid: 0000-0002-9008-7562
    affiliation: 1
affiliations:
 - name: Yoummday GmbH
   index: 1
date: 09 July 2022
bibliography: paper.bib
---

# Abstract
In speech processing, features extraction is essentially the estimation of a parametric representation of an input signal.
This is a key step in any audio based modeling and recognition process (e.g. speech recognition, sound classification, speaker authentication etc.).
There are several speech features to extract, such as the Linear Frequency Cepstral Coefficients (LFCC), Mel Frequency Cepstral Coefficients (MFCC), Linear Predictive Coding (LPC), and Constant-Q Cepstral Coefficients (CQCC) etc.
Each type of features has its own advantages and drawbacks (e.g. noise robustness, complexity, inter-components correlation etc.) that can directly affect the researched topic.
Unfortunately, existing libraries for extracting these features (e.g. python_speech_features [@python_speech_features:2020], SpeechPy [@speechpy:2018] and Bob [@bob:2017]) are limited and mostly focus on one extraction technique (e.g. MFCC), thus it is hard to find reliable implementations of other features extraction algorithms.
Consequently, this slows down the research and hinders the possibility of exploring, comparing or leveraging these different approaches against each other.
Hence, the need for **spafe**, a straightforward solution that unites all these different techniques in one python package.


This paper describes version 0.2.0 of spafe: a python package for audio features extraction based on the Numpy [@numpy:2020] and Scipy [@scipy:2019] libraries.
Spafe implements various features extraction techniques that can be used to solve a wide variety of recognition and classification tasks (speaker verification, spoken emotion recognition, spoken language identification etc.).
The paper provides a brief overview of the library’s structure, theory and functionalities.

# Introduction
Oftentimes, researchers find themselves constrained by the available tools.
This is the case of cepstral features based recognition processes, where most existing research relies on the Mel Frequency Cepstral Coefficents (MFCC) with very few exceptions.
To solve this, spafe is introduced.

The philosophy of spafe is keeping it simple, flexible and efficient in order to reach a wide range of developers and researchers.
Hence, spafe is written in python 3 and only depends on Numpy [@numpy:2020] and Scipy [@scipy:2019]. The library is heavily documented with the help of Sphinx and tested using Pytest.
Spafe supports **mono signals processing** and has been tested with different sampling rates (e.g. 8kHz, 16Khz, 44.1kHz, 48kHz etc.).

Scripts in spafe are divided into four major groups (see Figure 1):

\begin{table}[!h]
\begin{center}
    \begin{tabular}{l*{4}{l}r}
      - \textbf{fbanks}       & filter banks implementations.\\
      - \textbf{features}     & features extraction implementations.\\
      - \textbf{frequencies}  & frequencies based features extraction implementations.\\
      - \textbf{utils}        & helper functions for pre- \& post-processing and visualization etc. \\
    \end{tabular}
\end{center}
\end{table}

\begin{figure}[!h]
\begin{center}
      \includegraphics[width=13.5cm]{figures/spafe-structure.png}  
\end{center}
\caption{Structure of spafe.}
\end{figure}

# Implementation and theory
## Filter banks (`spafe/fbanks`)
A filter bank is defined as an array of band pass filters that splits the input signal into a set of analysis signals, each one carrying a single frequency sub-band of the original signal [@sarangi:2020; @penedo:2019]. Each band pass filter is centered at a different frequency, called center frequency. The center frequencies are evenly spaced over a specificied scaled frequencies range(e.g. bark scale, erb scale, mel scale etc.).
The bandwidths of the filters increase with the frequency, in order to duplicate the human hearing properties, which are characterized by a decreasing sensitivity at higher frequencies.
Within this context, spafe provides implementations for the following filter banks:
**Bark filter banks**, **Gammatone filter banks**, **Linear filter banks** and the **Mel filter banks**.

\begin{figure}[!h]
\begin{center}
      \includegraphics[width=13.5cm]{figures/bark_fbanks.png}  
\end{center}
\caption{Bark filter banks computed and visualized using spafe}
\end{figure}


## Features (`spafe/features`)
In an attempt to cover most audio features, spafe provides various frequency and cepstral domain features extraction algorithms, both filter bank-based and auto-regression-based. The following is a list of the available features extraction routines in the spafe python package:

\begin{table}[!h]
\begin{center}
    \begin{tabular}{l*{6}{l}r}
         - Bark Frequency Cepstral Coefﬁcients                 & BFCC  \\
         - Constant Q-transform Cepstral Coeﬃcients           & CQCC  \\
         - Gammatone Frequency Cepstral Coefﬁcients            & GFCC  \\
         - Linear Frequency Cepstral Coefﬁcients               & LFCC  \\
         - Linear Prediction Cepstral Coeﬃcients              & LPCC  \\
         - Mel Frequency Cepstral Coefﬁcients                  & MFCC  \\
         - Inverse Mel Frequency Cepstral Coefﬁcients          & IMFCC \\
         - Magnitude based Spectral Root Cepstral Coefficients & MSRCC \\
         - Normalized Gammachirp Cepstral Coefficients         & NGCC  \\
         - Power-Normalized Cepstral Coefficients              & PNCC  \\
         - Phase based Spectral Root Cepstral Coefficients     & PSRCC \\
         - Perceptual Linear Prediction Coefficents            & PLP   \\
         - Rasta Perceptual Linear Prediction Coefficents      & RPLP  \\
    \end{tabular}
\end{center}
\end{table}

The following figure provides a summary of the included features extraction algorithms and their detailed steps:

\begin{figure}[!h]
\begin{center}
      \includegraphics[width=14.5cm]{figures/features-extraction-algorithms.png}  
\end{center}
\caption{Features extraction algorithms in spafe}
\end{figure}

In addition to the previously mentioned features, spafe allows for computing the following spectrograms:
**Bark spectrogram**, **Cqt spectrogram**, **Erb spectrogram** and **Mel spectrogram**.

\begin{figure}[!h]
\begin{center}
      \includegraphics[width=14.5cm]{figures/melspectrogram.png}  
\end{center}
\caption{Mel spectrogram computed and visualized using spafe}
\end{figure}

## Frequencies (`spafe/frequencies`)
The frequencies modules in spafe focus specifically on the computation of dominant and fundamental frequencies.
A dominant frequency is per definition the frequency carrying the maximum energy among all frequencies of the spectrum [@rastislav:2013], whereas the fundamental frequency (often noted as $F_0$ ) is defined as the inverse of the period of a periodic signal [@cheveigne:2002].

\begin{figure}[!h]
\begin{center}
      \includegraphics[width=13cm]{figures/dominant_frequencies.png}  
\end{center}
\caption{Dominant frequencies computed and visualized using spafe}
\end{figure}

## Utils (`spafe/utils`)
The utils scripts, handle most of the input signal pre-processing steps including pre-emphasis, framing and windowing.
They also include all the conversion computations needed to convert Hertz frequencies to other frequencies scales.
On top of that, all features post-processing routines are in this group. This includes normalization, liftering, deltas computation and visualization.

# Conclusion
This paper introduced spafe, a python package for audio features extractions.
Spafe provides a unified solution for audio features extraction, that can help simplify and accelerate the research of various audio based recognition experiments.


# References
