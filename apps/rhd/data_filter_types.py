from enum import Enum

class UndersamplingFiltering(Enum):
    NONE = 0  # No undersampling (no filtering)
    NORMAL_BORDERLINE = 1
    NORMAL_DEFINITE = 2
    NORMAL_DEFINITE_BORDERLINE = 3
    NORMAL_DISEASECARRIER = 4

class DopplerFiltering(Enum):
    NONE = 5 # With or without doppler (no filtering)
    ONLY_WITHOUT_DOPPLER = 6
    ONLY_WITH_DOPPLER = 7