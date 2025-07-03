CHROM_SIZES = {
    "I": 230218,
    "II": 813184,
    "III": 316620,
    "IV": 1531933,
    "IX": 439888,
    "V": 576874,
    "VI": 270161,
    "VII": 1090940,
    "VIII": 562643,
    "X": 745751,
    "XI": 666816,
    "XII": 1078177,
    "XIII": 924431,
    "XIV": 784333,
    "XV": 1091291,
    "XVI": 948066,
}

HG38_CHROM_SIZES = {
    "chr7": 159345973,
}

nucleotide2onehot = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}

onehot2nucleotide = {tuple(v): k for k, v in nucleotide2onehot.items()}
