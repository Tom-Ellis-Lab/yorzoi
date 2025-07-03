import pandas as pd
import torch
from torch.utils.data import Dataset
from clex.dataset import GenomicDataset

class PromoterDataset(Dataset):
    upstream_polyT_777bp = "gcggcagaagaagtaacaaaggaacctagaggccttttgatgttagcagaattgtcatgcaagggctccctatctactggagaatatactaagggtactgttgacattgcgaagagcgacaaagattttgttatcggctttattgctcaaagagacatgggtggaagagatgaaggttacgattggttgattatgacacccggtgtgggtttagatgacaagggagacgcattgggtcaacagtatagaaccgtggatgatgtggtctctacaggatctgacattattattgttggaagaggactatttgcaaagggaagggatgctaaggtagagggtgaacgttacagaaaagcaggctgggaagcatatttgagaagatgcggccagcaaaactaaaaaactgtattataagtaaatgcatgtatactaaactcacaaattagagcttcaatttaattatatcagttattaccctatgcggtgtgaaataccgcacagatgcgtaaggagaaaataccgcatcaggaaattgtaagcgttaatattttgttaaaattcgcgttaaatttttgttaaatcagctcattttttaaccaataggccgaaatcggcaaaatcccttataaatcaaaagaatagaccgagatagggttgagtgttgttccagtttggaacaagagtccactattaaagaacgtggactccaacgtcaaagggcgaaaaaccgtctatcagggcgatggcccactacgtgaaccatcaccctaatcaagt"
    polyT_no_overhang = "gctagcaggaatgatgcaaaaggttcccgattcgaac"
    polyA_no_overhang = (
        "tcttaattaaaaaaagatagaaaacattaggagtgtaacacaagactttcggatcctgagcaggcaagataaacga"
    )
    YFP_connector = "aggcaaag"
    YFP = "atgtctaaaggtgaagaattattcactggtgttgtcccaattttggttgaattagatggtgatgttaatggtcacaaattttctgtctccggtgaaggtgaaggtgatgctacttacggtaaattgaccttaaaattgatttgtactactggtaaattgccagttccatggccaaccttagtcactactttaggttatggtttgcaatgttttgctagatacccagatcatatgaaacaacatgactttttcaagtctgccatgccagaaggttatgttcaagaaagaactatttttttcaaagatgacggtaactacaagaccagagctgaagtcaagtttgaaggtgataccttagttaatagaatcgaattaaaaggtattgattttaaagaagatggtaacattttaggtcacaaattggaatacaactataactctcacaatgtttacatcactgctgacaaacaaaagaatggtatcaaagctaacttcaaaattagacacaacattgaagatggtggtgttcaattagctgaccattatcaacaaaatactccaattggtgatggtccagtcttgttaccagacaaccattacttatcctatcaatctgccttatccaaagatccaaacgaaaagagagaccacatggtcttgttagaatttgttactgctgctggtattacccatggtatggatgaattgtacaaataa"
    YFP_term_connector = "ggcgcgccacttctaaataa"
    ADH1_terminator = "gcgaatttcttatgatttatgatttttattattaaataagttataaaaaaaataagtgtatacaaattttaaagtgactcttaggttttaaaacgaaaattcttattcttgagtaactctttcctgtaggtcaggttgctttctcaggtatagtatgaggtcgctcttattgaccacacctctaccgg"
    term_natmx6_connector = (
        "cagatccgctagggataacagggtaatatagatctgtttagcttgcctcgtccccgccgggtcacccggccagc"
    )
    natmx6 = "gacatggaggcccagaataccctccttgacagtcttgacgtgcgcagctcaggggcatgatgtgactgtcgcccgtacatttagcccatacatccccatgtataatcatttgcatccatacattttgatggccgcacggcgcgaagcaaaaattacggctcctcgctgcagacctgcgagcagggaaacgctcccctcacagacgcgttgaattgtccccacgccgcgcccctgtagagaaatataaaaggttaggatttgccactgaggttcttctttcatatacttccttttaaaatcttgctaggatacagttctcacatcacatccgaacataaacaaccatgggtaccactcttgacgacacggcttaccggtaccgcaccagtgtcccgggggacgccgaggccatcgaggcactggatgggtccttcaccaccgacaccgtcttccgcgtcaccgccaccggggacggcttcaccctgcgggaggtgccggtggacccgcccctgaccaaggtgttccccgacgacgaatcggacgacgaatcggacgacggggaggacggcgacccggactcccggacgttcgtcgcgtacggggacgacggcgacctggcgggcttcgtggtcgtctcgtactccggctggaaccgccggctgaccgtcgaggacatcgaggtcgccccggagcaccgggggcacggggtcgggcgcgcgttgatggggctcgcgacggagttcgcccgcgagcggggcgccgggcacctctggctggaggtcaccaacgtcaacgcaccggcgatccacgcgtaccggcggatggggttcaccctctgcggcctggacaccgccctgtacgacggcaccgcctcggacggcgagcaggcgctctacatgagcatgccctgcccctaatcagtactgacaataaaaagattcttgttttcaagaacttgtcatttgtatagtttttttatattgtagttgttctattttaatcaaatgttagcgtgatttatattttttttcgcctcgacatcatctgcccagatgcgaagttaagtgcgcagaaagtaatatcatgcgtcaatcgtatgtgaatgctggtcgctatactg"
    natmx6_cen_connector = (
        "ctgtcgattcgatactaacgccgccatccagtgtcgaaaacgagctcgaattcctgggtccttttc"
    )
    cen = "atcacgtgctataaaaataattataatttaaattttttaatataaatatataaattaaaaatagaaagtaaaaaaagaaattaaagaaaaaatagtttttgttttccgaagatgtaaaagactctagggggatcgccaacaaatactaccttttatcttgctcttcctgctctcaggtattaatgccgaattgtttcatcttgtctgtgtagaagaccacacacgaaaatcctgtgattttacattttacttatcgttaatcgaatgtatatctatttaatctgcttttcttgtctaataaatatatatgtaaagtacgctttttgttgaaattttttaaacctttgtttatttttttttcttcattccgtaactcttctaccttctttatttactttctaaaatccaaatacaaaacataaaaataaataaacacagagtaaattcccaaattattccatcattaaaagatacgaggcgcgtgtaagttacaggcaagcgatc"
    cen_downstream = "cgtccgatatcatcagatccactagtggcctatgcggccgcggatctgccggtctccctatagtgagtcgtattaatttcgataagccaggttaacctgcattaatgaatcggccaacgcgcggggagaggcggtttgcgtattgggcgctcttccgcttcctcgctcactgactcgctgcgctcggtcgttcggctgcggcgagcggtatcagctcactcaaaggcggtaatacggttatccacagaatcaggggataacgcaggaaagaacatgtgagcaaaaggccagcaaaaggccaggaaccgtaaaaaggccgcgttgctggcgtttttccataggctccgcccccctgacgagcatcacaaaaatcgacgctcaagtcagaggtggcgaaacccgacaggactataaagataccaggcgtttccccctggaagctccctcgtgcgctctcctgttccgaccctgccgcttaccggatacctgtccgcctttctcccttcgggaagcgtggcgctttctcatagctcacgctgtaggtatctcagttcggtgtaggtcgttcgctccaagctgggctgtgtgcacgaaccccccgttcagcccgaccgctgcgccttatccggtaactatcgtcttgagtccaacccggtaagacacgacttatcgccactggcagcagccactggtaacaggattagcagagcgaggtatgtaggcggtgctacagagttcttgaagtggtggcctaactacggctacactagaagaacagtatttggtatctgcgctctgctgaagccagttaccttcggaaaaagagttggtagctcttgatccggcaaacaaaccaccgctggtagcggtggtttttttgtttgcaagcagcagattacgcgcagaaaaaaaggatctcaagaagatcctttgatcttttctacggggtctgacgctcagtggaacgaaaactcacgttaagggattttggtcatgagattatcaaaaaggatcttcacctagatccttttaaattaaaaatgaagttttaaatcaatctaaagtatatatgagtaaacttggtctgacagttaccaatgcttaatcagtgaggcacctatctcagcgatctgtctatttcgttcatccatagttgcctgactccccgtcgtgtagataactacgatacgggagggcttaccatctggccccagtgctgcaatgataccgcgagacccacgttcaccggctccagatttatcagcaataaaccagccagccggaagggccgagcgcagaagtg"

    polyT_overhang = "TGCATTTTTTTCACATC".lower()
    polyA_overhang = "GGTTACGGCTGTT".lower()

    def __init__(self, promoter_seq_file: str):
        self.data = pd.read_csv(
            promoter_seq_file, sep="\t", header=None, names=["seq", "el"]
        )

    def construct_sequence(self, n80_with_overhang: str):
        n80_with_overhang = n80_with_overhang.lower()

        assert len(n80_with_overhang) == 110, (
            f"n80 with overhang is not 110bp long. {len(n80_with_overhang)}"
        )

        assert n80_with_overhang.startswith(PromoterDataset.polyT_overhang), (
            f"Seq doesn't start with polyT overhang! PolyT: {PromoterDataset.polyT_overhang} but got {n80_with_overhang[: len(PromoterDataset.polyT_overhang)]}"
        )
        assert n80_with_overhang.endswith(PromoterDataset.polyA_overhang), (
            f"Seq doesn't start with polyT overhang! PolyA: {PromoterDataset.polyA_overhang} but got {n80_with_overhang[-len(PromoterDataset.polyA_overhang) :]}"
        )

        full_seq = (
            PromoterDataset.upstream_polyT_777bp # 777bp
            + PromoterDataset.polyT_no_overhang # 37 bp 
            + n80_with_overhang # 110 bp
            + PromoterDataset.polyA_no_overhang # 76 bp
            + PromoterDataset.YFP_connector # 8 bp
            + PromoterDataset.YFP # 717 bp
            + PromoterDataset.YFP_term_connector
            + PromoterDataset.ADH1_terminator
            + PromoterDataset.term_natmx6_connector
            + PromoterDataset.natmx6
            + PromoterDataset.natmx6_cen_connector
            + PromoterDataset.cen
            + PromoterDataset.cen_downstream
        )

        assert len(full_seq) == 5000, (
            f"Sequence is not 5 Kb long. Seq len {len(full_seq)}"
        )

        return full_seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        construct_sequence = self.construct_sequence(self.data.iloc[idx]["seq"]).upper()

        return (
            torch.tensor(
                GenomicDataset.one_hot_encode(construct_sequence), dtype=torch.float32
            ),
            torch.tensor([self.data.iloc[idx]["el"]]),
        )
