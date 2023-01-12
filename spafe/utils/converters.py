"""

- Description : Frequency converters implementation.
- Copyright (c) 2019-2023 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
import numpy as np
from typing_extensions import Literal

# init vars
F0 = 0
FSP = 200 / 3
BARK_FREQ = 1000
BARK_PT = (BARK_FREQ - F0) / FSP
LOGSTEP = np.exp(np.log(6.4) / 27.0)
A = (1000 * np.log(10)) / (24.7 * 4.37)

ErbConversionApproach = Literal["Glasberg"]


def hz2erb(f: float, approach: ErbConversionApproach = "Glasberg") -> float:
    """
    Convert Hz frequencies to Erb as referenced in [Glasberg]_.

    Args:
        f      (float) : input frequency [Hz].
        approach (str) : conversion approach.
                         (Default is "Glasberg").

    Returns:
        (float): frequency in Erb [Erb].

    Note:
        Glasberg                (1990) :
            - :math:`fe = A . log_{10}(1 + f . 0.00437)`
            - :math:`f  = \\frac{10^{\\frac{fe}{A}} - 1}{0.00437}`

            where :math:`A = \\frac{1000 . log_{e}(10)}{24.7 . 4.37}`

            **!** might raise: RuntimeWarning: invalid value encountered in log10


    References:
        .. [Glasberg] : Glasberg B. R., and Moore B. C. J. "Derivation of
                        Auditory Filter Shapes from Notched-Noise Data." Hearing
                        Research. Vol. 47, Issues 1–2, 1990, pp. 103–138.

    Examples:
        .. plot::

            import matplotlib.pyplot as plt
            from spafe.utils.converters import hz2erb

            # generate freqs array -> convert freqs
            hz_freqs = [freq for freq in range(0, 8000, 10)]
            erb_freqs = [hz2erb(freq) for freq in hz_freqs]

            # visualize conversion
            plt.figure(figsize=(14,4))
            plt.plot(hz_freqs, erb_freqs)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Frequency (Erb)")
            plt.title("Hertz to Erb scale frequency conversion")
            plt.tight_layout()
            plt.show()
    """
    if approach == "Glasberg":
        return A * np.log10(1 + f * 0.00437)


def erb2hz(fe: float, approach: ErbConversionApproach = "Glasberg") -> float:
    """
    Convert Erb frequencies to Hz as referenced in [Glasberg]_.

    Args:
        fe      (float) : input frequency [Erb].
        approach  (str) : conversion approach.
                          (Default is "Glasberg").

    Returns:
        (float) : frequency in Hz [Hz].

    Note:
        Glasberg                (1990) :
            - :math:`fe = A . log_{10}(1 + f . 0.00437)`
            - :math:`f  = \\frac{10^{\\frac{fe}{A}} - 1}{0.00437}`

            where :math:`A = \\frac{1000 . log_{e}(10)}{24.7 . 4.37}`

            **!** might raise: RuntimeWarning: invalid value encountered in log10

    Examples:
        .. plot::

            import matplotlib.pyplot as plt
            from spafe.utils.converters import erb2hz

            # generate freqs array -> convert freqs
            erb_freqs = [freq for freq in range(0, 35, 1)]
            hz_freqs = [erb2hz(freq) for freq in erb_freqs]

            # visualize conversion
            plt.figure(figsize=(14,4))
            plt.plot(erb_freqs, hz_freqs)
            plt.xlabel("Frequency (Erb)")
            plt.ylabel("Frequency (Hz)")
            plt.title("Erb to Hertz frequency conversion")
            plt.tight_layout()
            plt.show()
    """
    if approach == "Glasberg":
        return (10 ** (fe / A) - 1) / 0.00437


BarkConversionApproach = Literal[
    "Wang", "Tjomov", "Schroeder", "Terhardt", "Zwicker", "Traunmueller"
]


def hz2bark(f: float, approach: BarkConversionApproach = "Wang") -> float:
    """
    Convert Hz frequencies to Bark as mentioned in [Carter]_ and [Traunmueller]_.

    Args:
        f      (float) : input frequency [Hz].
        approach (str) : conversion approach.
                         (Default is "Wang").

    Returns:
        (float): frequency in Bark [Bark].

    Note:
        Tjomov                (1971) :
            - :math:`fb = 6.7 . sinh^{-1}(\\frac{f+20}{600})`
            - :math:`f  = 600 . sinh(\\frac{fb}{6.7}) - 20`

        Schroeder             (1977) :
            - :math:`fb = 7 . sinh^{-1}(\\frac{f}{650})`
            - :math:`f  = 650 . sinh(\\frac{fb}{7})`

        Terhardt              (1979) :
            - :math:`fb = 13.3*tan^{-1}(\\frac{0.75 . f}{1000})`
            - :math:`f  = (1000/0.75)*tan(\\frac{fb}{13})`

        Zwicker & Terhardt    (1980) :
            - :math:`fb = 8.7 + 14.2 . log10(\\frac{f}{1000})`
            - :math:`f  = 10^{(\\frac{fb-8.7}{14.2} + 3)}`

        Traunmueller          (1983) :
            - :math:`fb = (\\frac{26.81*f}{1+1960}) - 0.53`
            - :math:`f  = 1960 . (\\frac{fb+0.53}{26.28-fb})`

        Wang, Sekey & Gersho  (1992) :
            - :math:`fb = 6 . sinh^{-1}(\\frac{f}{600})`
            - :math:`f  = 600 . sinh(\\frac{fb}{6})`

    Examples:
        .. plot::

            import matplotlib.pyplot as plt
            from spafe.utils.converters import hz2bark

            # generate freqs array -> convert freqs
            hz_freqs = [freq for freq in range(0, 8000, 10)]
            bark_freqs = [hz2bark(freq) for freq in hz_freqs]

            # visualize conversion
            plt.figure(figsize=(14,4))
            plt.plot(hz_freqs, bark_freqs)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Frequency (Bark)")
            plt.title("Hertz to Bark scale frequency conversion")
            plt.tight_layout()
            plt.show()
    """
    if approach == "Wang":
        return 6 * np.arcsinh(f / 600)
    elif approach == "Tjomov":
        return 6.7 * np.arcsinh((f + 20) / 600)
    elif approach == "Schroeder":
        return 7 * np.arcsinh(f / 650)
    elif approach == "Terhardt":
        return 13.3 * np.arctan((f * 0.75) / 1000)
    elif approach == "Zwicker":
        return 8.7 + 14.2 * np.log10(f / 1000)
    elif approach == "Traunmueller":
        return ((26.28 * f) / (1 + 1960)) - 0.53
    else:
        return 6 * np.arcsinh(f / 600)


def bark2hz(fb: float, approach: BarkConversionApproach = "Wang") -> float:
    """
    Convert Bark frequencies to Hz as mentioned in [Carter]_ and [Traunmueller]_.

    Args:
        fb     (float) : input frequency [Bark].
        approach (str) : conversion approach.
                         (Default is "Wang").
    Returns:
        (float) : frequency in Hz [Hz].

    Note:
        Tjomov                (1971) :
            - :math:`fb = 6.7 . sinh^{-1}(\\frac{f+20}{600})`
            - :math:`f  = 600 . sinh(\\frac{fb}{6.7}) - 20`

        Schroeder             (1977) :
            - :math:`fb = 7 . sinh^{-1}(\\frac{f}{650})`
            - :math:`f  = 650 . sinh(\\frac{fb}{7})`

        Terhardt              (1979) :
            - :math:`fb = 13.3*tan^{-1}(\\frac{0.75 . f}{1000})`
            - :math:`f  = (1000/0.75)*tan(\\frac{fb}{13})`

        Zwicker & Terhardt    (1980) :
            - :math:`fb = 8.7 + 14.2 . log10(\\frac{f}{1000})`
            - :math:`f  = 10^{(\\frac{fb-8.7}{14.2} + 3)}`

            *!* might raise RuntimeWarning: divide by zero encountered in log10

        Traunmueller          (1983) :
            - :math:`fb = (\\frac{26.81*f}{1+1960}) - 0.53`
            - :math:`f  = 1960 . (\\frac{fb+0.53}{26.28-fb})`

        Wang, Sekey & Gersho  (1992) :
            - :math:`fb = 6 . sinh^{-1}(\\frac{f}{600})`
            - :math:`f  = 600 . sinh(\\frac{fb}{6})`

    References:
        .. [Carter] Carter, P., "Sonification seminar – 10/9/03", CCRMA.Stanford.edu.,
                    https://ccrma.stanford.edu/courses/120-fall-2003/lecture-5.html
        .. [Traunmueller] Traunmueller, H. (1990). Analytical expressions for the tonotopic sensory scale.
                         The Journal of the Acoustical Society of America, 88(1), 97–100. doi:10.1121/1.399849

    Examples:
        .. plot::

            import matplotlib.pyplot as plt
            from spafe.utils.converters import bark2hz

            # generate freqs array -> convert freqs
            bark_freqs = [freq for freq in range(0, 80, 5)]
            hz_freqs = [bark2hz(freq) for freq in bark_freqs]

            # visualize conversion
            plt.figure(figsize=(14,4))
            plt.plot(bark_freqs, hz_freqs)
            plt.xlabel("Frequency (Bark)")
            plt.ylabel("Frequency (Hz)")
            plt.title("Bark to Hertz frequency conversion")
            plt.tight_layout()
            plt.show()
    """
    if approach == "Wang":
        return 600 * np.sinh(fb / 6)
    elif approach == "Tjomov":
        return 600 * np.sinh(fb / 6.7) - 20
    elif approach == "Schroeder":
        return 650 * np.sinh(fb / 7)
    elif approach == "Terhardt":
        return (1000 / 0.75) * np.tan(fb / 13)
    elif approach == "Zwicker":
        return 10 ** (((fb - 8.7) / 14.2) + 3)
    elif approach == "Traunmueller":
        return 1960 * (
            (__traunmueller_helper(fb) + 0.53) / (26.28 - __traunmueller_helper(fb))
        )
    else:
        return 600 * np.sinh(fb / 6)


def __traunmueller_helper(fi: float) -> float:
    """
    Helper funtion for the Traunmueller approach.
    """
    if fi < 2:
        return (fi - 0.3) / (0.85)
    elif fi > 20.1:
        return (fi + 4.422) / 1.22
    else:
        return fi


MelConversionApproach = Literal["Oshaghnessy", "Lindsay"]


def hz2mel(f: float, approach: MelConversionApproach = "Oshaghnessy") -> float:
    """
    Convert a value in Hertz to Mels [Oshaghnessy]_, [Beranek]_ and [Lindsay]_.

    Args:
        f      (float) : input frequency [Hz].
        approach (str) : conversion approach.
                         (Default is "Oshaghnessy").
    Returns:
        (float) : frequency in Mel scale [Mel].

    Note:
        Oshaghnessy                (1987) :
            - :math:`fm = 2595 . log_{10}(1 + \\frac{f}{700})`
            - :math:`f  = 700 . (10^{(\\frac{fm}{2595} - 1)}`

        Beranek                (1987) :
            - :math:`fm = 1127 . log_{e}(1 + \\frac{f}{700})`
            - :math:`f  = 700 . exp(\\frac{fm}{1127} - 1)`

            * Both previous equations correspond to each other.

        Lindsay                    (1977) :
            - :math:`fm = 2410 . log_{10}(1 + \\frac{f}{625})`
            - :math:`f  = 625 . (10^{(\\frac{fm}{2410} - 1)}`

    References:
        .. [Oshaghnessy] : O'Shaghnessy, Douglas. Speech Communication: Human
                          and Machine. Reading, MA: Addison-Wesley Publishing Company, 1987.

        .. [Beranek] : Beranek L.L. Acoustic Measurements, (1949) New York: Wiley.

        .. [Lindsay] : Lindsay, Peter H.; & Norman, Donald A. (1977).
                       Human information processing: An introduction to psychology
                       (2nd ed.). New York: Academic Press.

    Examples:
        .. plot::

            import matplotlib.pyplot as plt
            from spafe.utils.converters import hz2mel

            # generate freqs array -> convert freqs
            hz_freqs = [freq for freq in range(0, 8000, 100)]
            mel_freqs = [hz2mel(freq) for freq in hz_freqs]

            # visualize conversion
            plt.figure(figsize=(14,4))
            plt.plot(hz_freqs, mel_freqs)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Frequency (Mel)")
            plt.title("Hertz to Mel frequency conversion")
            plt.tight_layout()
            plt.show()
    """
    return {
        "Oshaghnessy": 2595 * np.log10(1 + f / 700.0),
        "Lindsay": 2410 * np.log10(1 + f / 625),
    }[approach]


def mel2hz(fm: float, approach: MelConversionApproach = "Oshaghnessy") -> float:
    """
    Convert a value in Mels to Hertz

    Args:
        fm     (float) : input frequency [Mel].
        approach (str) : conversion approach.
                         (Default is "Wang").
    Returns:
        (float) : frequency in Hz [Hz].

    Examples:
        .. plot::

            import matplotlib.pyplot as plt
            from spafe.utils.converters import mel2hz

            # generate freqs array -> convert freqs
            mel_freqs = [freq for freq in range(0, 8000, 100)]
            hz_freqs = [mel2hz(freq) for freq in mel_freqs]

            # visualize conversion
            plt.figure(figsize=(14,4))
            plt.plot(mel_freqs, hz_freqs)
            plt.xlabel("Frequency (Mel)")
            plt.ylabel("Frequency (Hz)")
            plt.title("Mel to Hertz frequency conversion")
            plt.tight_layout()
            plt.show()
    """
    return {
        "Oshaghnessy": 700 * (10 ** (fm / 2595.0) - 1),
        "Lindsay": 625 * (10 ** (fm / 2410) - 1),
    }[approach]
