""" Analysis of synthetic peptides """
import os
import re
import collections
import itertools
import json
import typing
import numpy as np
from operator import attrgetter
from _bisect import bisect_left

from pepfrag import constants, Peptide, ModSite
from rPTMDetermine.peptide_spectrum_match import PSM
from rPTMDetermine.readers import PTMDB


ptmdb = PTMDB()

Feature_names = (
    "NumPeaks", "TotInt", "PepMass", "Charge", "FracIon", "FracIonInt",
    "NumSeriesbm", "NumSeriesym", "NumIona", "NumIonynl", "NumIonbnl",
    "FracIonIntb_c1", "FracIonIntb_c2", "FracIonInty_c1", "FracIonInty_c2",
    "FracIon20pc", "NumIonb", "NumIony", "FracIonInty", "FracIonIntb",
    "MatchScore", "SeqTagm"
)

target_pairs = {
    "Kmod_Biotinyl": "Kmod_Biotin", "Kmod_Propionyl": "Kmod_Propion",
    "Kmod_Ubiquitinyl": "Kmod_Glygly"
}

SynMatch = collections.namedtuple(
    "SynMatch", ["seq", "mods", "charge", "num_ions", "prec_mz", "num_seqtags",
                 "max_tag", "ion_index", "delta_mass", "nterm_left",
                 "cterm_left"],
    defaults=[None] * 11
)

SeqCorrect = collections.namedtuple(
    "SeqCorrect", ["rm_idx", "rt_seq", "ist_idx", "ist_seq", "mods"],
    defaults=[None] * 5
)

ProcMatch = collections.namedtuple(
    "ProcMatch", ["seq_prefix", "mass_prefix", "mods_prefix", "seq_proc",
                  "mods_proc", "seq_suffix", "mass_suffix", "mods_suffix"],
    defaults=[None] * 8
)

TagPrecursor = collections.namedtuple(
    "TagPrecursor",
    ["mz_tag", "pmass", "length", "seq_term", "ion_type", "mods_tag",
     "pseq", "index", "tag", "pmods", "index_tag"],
    defaults=[None] * 11
)


def mix(c: str, s: str, p: str = ''):
    """
    Insert characters c into all places in string s
    This is the answer in StackOverFlow:
    https://stackoverflow.com/questions/63488042/python-fast-insert-multiple-characters-into-all-possible-places-of-string/63488696#63488696
    Thanks to superb rain.
    """
    return c and s and mix(c[1:], s, p+c[0]) + mix(c, s[1:], p+s[0]) or [p + c + s]


def _peptide_mass(seq: str, mods: typing.List[ModSite]):
    """ Calculate peptide mass. """
    m = sum(constants.AA_MASSES[a].mono for a in seq)
    if mods:
        m += sum(mod.mass for mod in mods)
    return m + constants.FIXED_MASSES["H2O"]


def _common_substr(seq1: str, seq2: str) -> typing.List[str]:
    """ Longest common substring between two sequences. """
    prec_seq = np.array(list(seq2))
    # number of characters in sequence
    n = prec_seq.size
    # initializations, only use last and current arrays for updating
    # they are nested lists with inner list of tuples (i, j):
    # index of common residue in (seq1, seq2)
    S: typing.List[typing.List[tuple]] = [[] for i in range(n)]
    S0: typing.List[typing.List[tuple]] = [[] for i in range(n)]
    # the first element in first sequence
    ix, = np.where(prec_seq == seq1[0])
    for i in ix:
        S0[i].append((0, i))

    # search common substrings
    str_pre, str_next, common_str_index = ix.tolist(), [], []
    for i in range(1, len(seq1)):
        ix, = np.where(prec_seq == seq1[i])
        if ix.size > 0:
            if ix[0] == 0:
                S[0].append((i, 0))
                ix = ix[1:]
            for j in ix:
                S[j] = S0[j - 1] + [(i, j)]
                if len(S[j]) > 1:
                    str_next.append(j)
        # store common substrings
        if str_pre:
            common_str_index += [
                S0[j] for j in set(str_pre) - set(j - 1 for j in str_next)
                if S0[j] not in common_str_index and S0[j]
            ]
        # backup current match as previous match for the next and clear
        # current match
        S0, str_pre = S.copy(), str_next.copy()
        S, str_next = [[] for i in range(n)], []
    # the last element
    if str_pre:
        common_str_index += [S0[j] for j in str_next
                             if S0[j] not in common_str_index and S0[j]]

    return common_str_index


class _Spectrum:
    """ Composition from psm object """
    def __init__(self,
                 spectrum: np.ndarray,
                 precursor_mz: float = None):
        self.spectrum = spectrum
        self.precursor_mz = precursor_mz
        self.sequence_tags = None
        self.peak_index = None

    def denoise(self,
                num_retain: int = 6,
                noise: float = 0.005) -> np.ndarray:
        """
        Denoise mass spectrum based on the rule 6 top peaks in
        100 Da window.
        """
        peaks = self.spectrum
        # remove very low intensity peaks
        peaks = peaks[peaks[:, 1] / peaks[:, 1].max() >= noise]
        deisotope_tol = 0.2
        n_windows = int((peaks[-1][0] - peaks[0][0]) / 100.) + 1

        i0, npeaks = 0, peaks.shape[0]
        denoised_peaks = []
        # denoise spectrum
        for win in range(n_windows):
            # upper limit mz in current window
            max_mz = peaks[0][0] + (win + 1) * 100
            for i in range(i0, npeaks):
                if peaks[i][0] >= max_mz:
                    break
            if i == i0:
                continue

            # reset the starting peak index for next window.
            if i == npeaks - 1 and peaks[i][0] < max_mz:
                i = npeaks

            # deisotoping
            sub_peaks = self.deisotope(peaks[i0:i], tol=deisotope_tol)
            # sort peaks based on intensities
            sub_peaks = sub_peaks[sub_peaks[:, 1].argsort()[::-1]]
            denoised_peaks += sub_peaks[:num_retain].tolist()

            # reset the starting index
            i0 = i

        denoised_peaks = np.array(denoised_peaks)
        self.spectrum = denoised_peaks[denoised_peaks[:, 0].argsort()]

        return self.spectrum

    def deisotope(self, peaks: np.ndarray, tol: float = 0.2) -> np.ndarray:
        """
        Deisotope peaks.

        Args:
            peaks: Mass spectral peaks for deisotoping.
            tol: Tolerance in dalton to detect isotope peaks.

        Returns:
            De-isotoped peaks

        """
        if peaks.shape[0] == 1:
            return peaks

        # deisotoping
        deiso_peaks, rm_index = [], set()
        for i, peak in enumerate(peaks):
            if i in rm_index:
                continue

            # iterate through peaks to detect isotopes by assuming
            # different charge states
            iso_index = []
            for c in [1, 2]:
                peak0, j0, iso_index_c = peak, i, [i]
                while True:
                    has_isotope = False
                    for j, peak1 in enumerate(peaks[j0:]):
                        # TODO: use similarity comparison to detect
                        # isotopic distribution.
                        if (abs((peak1[0] - peak0[0]) * c - 1) <= tol
                                and 1 - peak1[1] / peak0[1] >= 0.2):
                            iso_index_c.append(j0 + j)
                            has_isotope = True
                            break
                    if not has_isotope:
                        break
                    j0 += j + 1
                    peak0 = peak1
                # maximal length as the detected isotopes
                if len(iso_index_c) > len(iso_index):
                    iso_index = iso_index_c

            # remove isotopic peaks
            rm_index.update(iso_index)
            deiso_peaks.append(peaks[iso_index[0]])

        return np.array(deiso_peaks)

    def extract_sequence_tags(self, tol: float = 0.2) -> None:
        """ Search tags corresponding to synthetic peptides. """
        # pairwise m/z differences
        diff_mz = self.spectrum[:, 0] - self.spectrum[:, 0][:, np.newaxis]
        # amino acid mass
        aas = "".join(sorted(constants.AA_MASSES.keys()))
        aam = np.array([constants.AA_MASSES[a].mono for a in aas])
        aam = aam[:, np.newaxis]
        aa_min, aa_max = aam.min(), aam.max()
        # get sequence tags from the spectrum
        seq_tags, index_tags, assigned = [], [], set()
        for i in range(self.spectrum.shape[0]):
            if i in assigned:
                continue

            # search tags
            alive_ix, alive_seqs = [[i]], [[]]
            while True:
                curr_ix, curr_seq = [], []
                for kx, aax in zip(alive_ix, alive_seqs):
                    # get next residue
                    k = kx[-1]
                    df_k = diff_mz[k][k+1:]
                    jx, = np.where(
                        (df_k <= aa_max + tol) & (df_k >= aa_min - tol)
                    )

                    # filter the m/z differences by min and max AA mass
                    ix, tags = [], []
                    if jx.size > 0:
                        aa_diff = np.absolute(df_k[jx] - aam)
                        for aa, diff in zip(aas, aa_diff):
                            _ix = jx[diff <= tol]
                            if _ix.size > 0:
                                ix += _ix.tolist()
                                tags += [aa] * _ix.size
                    if ix:
                        curr_ix += [kx + [j + k + 1] for j in ix]
                        curr_seq += [aax + [a] for a in tags]
                    elif len(aax) > 0:
                        seq_tags.append("".join(aax))
                        index_tags.append(kx)

                # if no new residue is found, stop the search
                if not curr_ix:
                    break
                alive_ix, alive_seqs = curr_ix, curr_seq

            # assigned indices
            assigned.update(itertools.chain(*index_tags))

        self.sequence_tags = seq_tags
        self.peak_index = index_tags

    def match_sequence(self, seq: str, mods: typing.List[ModSite]):
        """ Match sequence. """
        # parse modifications to separate them into three parts
        self._parse_modification(mods)
        # precursor mass
        pmass = _peptide_mass(seq, mods)

        # match seq by common subsequences
        prec_tag, assigned = [], set()
        # matches as b ions
        index_b = [(i, "b") for i, tag in enumerate(self.sequence_tags)
                   if any(tag[i:i+3] in seq for i in range(len(tag) - 2))]
        # matches as y ions
        seq2 = seq[::-1]
        index_y = [(i, "y") for i, tag in enumerate(self.sequence_tags)
                   if any(tag[i:i+3] in seq2 for i in range(len(tag) - 2))]
        # construct matches
        for i, yb_type in index_b + index_y:
            prec_tag_c = self._get_tag_precursor(
                seq, self.sequence_tags[i], ion_type=yb_type
            )
            # unique matches
            pix = self.peak_index[i]
            for t in prec_tag_c:
                mz = self.spectrum[pix[t.index_tag]][0]
                if (mz, t.seq_term, t.ion_type, t.index, t.length) in assigned:
                    continue
                prec_tag.append(t._replace(mz_tag=mz, pmass=pmass, pmods=mods))
                assigned.add((mz, t.seq_term, t.ion_type, t.index, t.length))

        return prec_tag

    def _get_tag_precursor(self, seq: str, tag: str,
                           ion_type: typing.Optional[str] = None):
        """
        Get fragment precursor containing tags.

        Args:
            seq: Parent sequence.
            ion_type: Type of fragment ion.
            pmass: Precursor mass.

        Returns:
            List of objects

        """
        seq_m = seq if ion_type == "b" else seq[::-1]
        ns = len(seq)
        mod_nterm, mod_cterm, mods_ints = self._parsed_mods
        # common subsequences
        precursors = []
        subseqs = _common_substr(tag, seq_m)
        for sub_index in subseqs:
            n = len(sub_index)
            # the number of tags should be larger than 2.
            if n <= 2:
                continue
            # index of common subseq starting and ending
            (_, j0), (i1, j1) = sub_index[0], sub_index[-1]
            # fragment precursors, subsequence and modifications
            term = seq_m[:j0+n]
            if ion_type == "y":
                term, j1 = term[::-1], ns - j1 - 1
                mods = (mod_cterm + [m._replace(site=int(m.site-j1))
                                     for m in mods_ints if m.site > j1])
            else:
                mods = mod_nterm + [m for m in mods_ints if m.site <= j0+1]
            # save the information
            seq_frag = {
                "seq_term": term, "length": n, "tag": sub_index,
                "mods_tag": mods, "pseq": seq, "ion_type": ion_type,
                "index_tag": i1 + 1, "index": j1
            }
            precursors.append(TagPrecursor(**seq_frag))

        return precursors

    def _parse_modification(self, mods: ModSite = None):
        """ Parse modifications. """
        # parse modifications
        mod_nterm, mod_cterm, mods_ints = [], [], []
        if mods is not None:
            for _mod in mods:
                if isinstance(_mod.site, int):
                    mods_ints.append(_mod)
                elif _mod.site == "nterm":
                    mod_nterm.append(_mod)
                else:
                    mod_cterm.append(_mod)
        self._parsed_mods = (mod_nterm, mod_cterm, mods_ints)


class CorrectSynthesisMatch:
    """ Correct false positive matches using synthetic peptides. """
    def __init__(self, validator, tol=0.1):
        # load targets
        self._get_targets()
        # load synthetic peptides
        self._get_synthetic_peptides()
        # load artifacts
        self._get_artifact()
        # validator
        self.validator = validator
        self.tol = tol

    def correct_psms(self, psm: PSM):
        """ Identify the mass spectrum from peptide synthesis error. """
        # match to synthetic peptides
        # synthetic peptide targets
        syn_peps = self.syn_peps[self._parse_raw_id(psm.data_id)]

        # annotate by synthetic peptides to get isobarics
        candidates = self._annotate_by_synthetic_peptides(psm, syn_peps)

        # matches to synthetic peptides
        matches = self._match_syn_peptide(psm, syn_peps)

        # refine the matches
        if matches:
            for match in matches:
                candidates += self._correct_psm(match)

        if not candidates:
            # sequence tag search
            tag_matches = self._sequence_tag_search(psm, syn_peps)
            if tag_matches is None:
                return None
            # correct the mass of tag precursors
            pmz_eu = psm.spectrum.prec_mz - constants.FIXED_MASSES["H"]
            for match, tag in tag_matches:
                candidates_tag = self._correct_psm(match)
                # correct the mass of the rest of the peptide
                for pep in candidates_tag:
                    m2 = self._tag_match_correct(pep, tag)
                    dm = [(m2.delta_mass - pmz_eu * c, c) for c in range(2, 5)]
                    candidates += self._correct_psm(m2._replace(delta_mass=dm))

        if candidates:
            return self._validate_candidates(psm, candidates)

        return None

    def _annotate_by_synthetic_peptides(self, psm, syn_peptides):
        """ Annotate the spectrum by corresponding synthetic peptide """
        mz = psm.spectrum.prec_mz
        pep_candidates = []
        for seq, mods in syn_peptides:
            for c in range(2, 5):
                pep = Peptide(seq, c, mods)
                mz_ = pep.mass / c + constants.FIXED_MASSES["H"]
                if abs(mz_ - mz) <= self.tol:
                    pep_candidates.append(pep)

            # consider artificial modifications too
            pep_candidates += self._add_mods(seq, mods, mz)
            pep_candidates += self._add_residue(seq, mods, mz)

        return pep_candidates

    @staticmethod
    def _match_syn_peptide(psm, syn_peptides, match=SynMatch):
        """ Match synthetic peptide forcely. """
        pmz_neu = psm.spectrum.prec_mz - constants.FIXED_MASSES["H"]
        # assign mass spectra using synthetic peptides
        matches = []
        for seq, mods in syn_peptides:
            # peptide project
            pep = Peptide(seq, 2, mods)
            mch = PSM(psm.data_id, psm.spec_id, pep, psm.spectrum)
            annotates, _ = mch.denoise_spectrum()

            # get annotations
            n, seq_ion_index = 0, collections.defaultdict(list)
            for ion, (_, ion_index) in annotates.items():
                if ion[0] in "yb" and "-" not in ion and ion.endswith("[+]"):
                    n += 1
                    seq_ion_index[ion[0]].append(ion_index)
            if n == 0:
                continue

            # maximum length of seq tags and the maximum ion index
            n_tag, max_ion_index = [], {"y": 0, "b": 0}
            for ion in seq_ion_index.keys():
                nion = len(seq_ion_index[ion])
                if nion > 1:
                    index_diff, i0 = np.diff(sorted(seq_ion_index[ion])), -1
                    tx, = np.where(index_diff > 1)
                    for i in tx:
                        n_tag.append(i - i0)
                        if i - i0 > 1:
                            max_ion_index[ion] = seq_ion_index[ion][i + 1]
                        i0 = i
                    # the end of the array
                    if nion - i0 > 2:
                        n_tag.append(nion - i0 - 1)
                        max_ion_index[ion] = max(seq_ion_index[ion])
                else:
                    n_tag.append(len(seq_ion_index[ion]))
                    if len(seq_ion_index[ion]) == 1:
                        max_ion_index[ion] = seq_ion_index[ion][0]

            # if number of ions in a sequence tag is gt 3 or more than
            # two sequence tags having number of ions equaling to 3
            if max(n_tag) >= 4 or n_tag.count(3) >= 2:
                # delta mass
                dms = [(pep.mass - pmz_neu * c, c) for c in range(2, 5)]
                # whether Terminus is tended to be modified
                matches.append(
                    match(seq=seq, mods=mods, charge=psm.charge, num_ions=n,
                          num_seqtags=n_tag, max_tag=max(n_tag),
                          ion_index=max_ion_index, delta_mass=dms,
                          nterm_left=max_ion_index["b"] == 0,
                          cterm_left=max_ion_index["y"] == 0)
                )

        return matches

    def _correct_psm(self, match):
        """ Correct PSM. """
        # sparated match to consider subsequence only
        sepm = self._parse_matches(match)
        # get combinations and artifacts
        combs, combs_add, artifs, rp, mod_dict = self._restrict_refs(
            sepm.seq_proc, sepm.mods_proc,
            nterm=match.nterm_left, cterm=match.cterm_left
        )
        s = "".join(rp)
        # get the combinations of error loss
        candidates, unique_peps = [], set()
        for dm, c in match.delta_mass:
            if abs(dm) <= self.tol:
                candidates.append(Peptide(match.seq, c, match.mods))
                continue

            # all possible corrections
            corrects = self._correct_mass(
                s, dm, combs, combs_add, artifs
            )
            # correction for removing residues
            for corr in corrects:
                seq_c = "".join(
                    mod_dict[r]["res"] if r in mod_dict else r for r in s
                )
                # reset modifications
                mods_c = []
                for i, r in enumerate(s):
                    if r in mod_dict:
                        mods_c.append(ModSite(mod_dict[r]["mass"],
                                              i+1, mod_dict[r]["mod"]))
                # set up modification sites from corrections.
                if corr.mods is not None:
                    mods_c += corr.mods

                seq, mods = self._reconstruct_peptide(sepm, seq_c, mods_c)
                # if the peptide has been identified, ignore it.
                pep = self._combine_mods(seq, mods)
                if pep not in unique_peps:
                    candidates.append(Peptide(seq, c, mods))
                    unique_peps.add(pep)

        return candidates

    def _correct_mass(self, seq: str, dm: float, res_combs: dict,
                      res_combs_add: dict, artifacts: dict):
        """
        Match artifact modifications and synthesis error.

        Args:
            seq: Sequence for correction.
            dm: Delta mass.
            res_combs: Combinations of residues for removing.
            res_combs_add: Combinations of residues as additional
                           residues for the correction.
            artifacts: Artifacts.

        Note:
            In this correction, removal of residues, addition of
            another residues, and then addition of chemical
            modifications are not considered. Maybe do it later.

        """
        corrects = []
        # additions
        add_res, res_mass = res_combs_add["residues"], res_combs_add["mass"]
        add_mod, mod_mass = artifacts["mods"], artifacts["mass"]
        # match residue combinations
        for i in range(max(res_combs.keys()) + 1):
            # i == 0 indicates no residue removed
            rm_index_seq = res_combs[i]["rm_index"] if i > 0 else [[]]
            mass_i = res_combs[i]["mass"] if i > 0 else [0.]

            if dm > 0 and i > 0:
                # remove residues from subseq in candidates
                ix, = np.where(np.absolute(mass_i - dm) <= self.tol)
                for j in ix:
                    seq2 = self._remove_mass(seq, rm_index_seq[j])
                    if seq2 is not None:
                        corrects.append(
                            SeqCorrect(rm_idx=rm_index_seq[j], rt_seq=seq2)
                        )

            # further combination of residues and modifications
            for rm_ix, m1 in zip(rm_index_seq, mass_i):
                # remove residues
                seq2 = self._remove_mass(seq, rm_ix)
                if seq2 is None:
                    continue
                # compensates from other residue combinations
                ix, = np.where(np.absolute(m1 - res_mass - dm) <= self.tol)
                r1_set = set(seq[i] for i in rm_ix)
                for j in ix:
                    if not r1_set & set(add_res[j]):
                        corrects += self._add_mass(seq2, rm_ix, add_res[j])
                # compensates from modifications
                ix = np.absolute(m1 - mod_mass - dm) <= self.tol
                for mod, m in zip(add_mod[ix], mod_mass[ix]):
                    corrects += self._add_mass(
                        seq2, ix, mod, seq_type="mod", mass=m
                    )

        return corrects

    @staticmethod
    def _remove_mass(seq, rm_ix):
        """ Remove residues rseq from seq. """
        if not rm_ix:
            return seq

        if len(seq) <= len(rm_ix):
            return None

        # remove residues based on the input index
        n = len(seq) - 1
        rm_ix = [0, *rm_ix, n if rm_ix[-1] == n else None]
        return "".join(seq[i+1:j] for i, j in zip(rm_ix[:-1], rm_ix[1:]))

    @staticmethod
    def _add_mass(seq, rm_index, add_seq, seq_type="seq", mass=None):
        """ Add residues to seq. """
        # simply insert add_seq to seq
        if seq_type == "seq":
            seqs_adds = mix(add_seq, seq)
            return [SeqCorrect(rm_idx=rm_index, ist_seq=add_seq, rt_seq=x)
                    for x in seqs_adds]

        # add modifications to seq
        elif seq_type == "mod":
            m, t = add_seq[0], add_seq[1]
            # sites
            if t not in constants.AA_MASSES:
                return [SeqCorrect(rm_idx=rm_index, rt_seq=seq,
                                   mods=[ModSite(mass, t, m)])]
            else:
                # matches
                return [
                    SeqCorrect(rm_idx=rm_index, rt_seq=seq,
                               mods=[ModSite(mass, i+1, m)])
                    for i, r in enumerate(seq) if r == t
                ]

    def _add_mods(self, seq, mods, mz):
        """ Add artificial modifications to sequence. """
        candidates = []
        # calculate mass
        pmass = _peptide_mass(seq, mods)
        # get possible modified sites
        unmod_res = [(i+1, r) for i, r in enumerate(seq)
                     if not any(mod.site == i+1 for mod in mods)]
        # consider terminal modifications too
        if not any(m.site == "nterm" for m in mods):
            unmod_res.insert(0, (0, "nterm"))
        if not any(m.site == "cterm" for m in mods):
            unmod_res.append((len(seq), "cterm"))
        unmod_res = [r for r in unmod_res if r[1] in self.artifacts]

        # re-organize the artifacts
        artifacts, masses = [], []
        for j, r in unmod_res:
            artifacts.extend([(j, mod, m) for mod, m in self.artifacts[r]])
            masses.extend([m for _, m in self.artifacts[r]])

        n, n1 = len(seq), len(masses)

        # artifact combinations
        artifacts += list(itertools.combinations(artifacts, 2))
        masses += [m1 + m2 for m1, m2 in itertools.combinations(masses, 2)]
        masses = np.array(masses)

        # iterate through all possibilities
        for c in range(2, 5):
            mass = (mz - constants.FIXED_MASSES["H"]) * c
            dm = mass - pmass
            # do all possible combinations
            ix, = np.where(np.absolute(dm - masses) <= self.tol)
            for i in ix:
                mod_x = []
                ax = [artifacts[i]] if i < n1 else artifacts[i]
                for j, name, m in ax:
                    site = "nterm" if j == 0 else "cterm" if j == n - 1 else j
                    mod_x.append(ModSite(m, site, name))
                candidates.append(Peptide(seq, c, mod_x + mods))

        return candidates

    def _add_residue(self, seq, mods, mz):
        """ Add residues to sequence. """
        pmass = _peptide_mass(seq, mods)
        mz_neutral = mz - constants.FIXED_MASSES["H"]
        # re-set the sequence by replacing the modified residue by number
        pre_mods, seq_x, mods_bk = [], seq, {}
        for i, mod in enumerate(mods):
            if isinstance(mod.site, str):
                pre_mods.append(mod)
            else:
                j = mod.site-1
                seq_x = f"{seq_x[:j]}{i}{seq_x[j+1:]}"
                mods_bk[f"{i}"] = {"n": mod.mod, "r": seq[j], "m": mod.mass}

        # added masses
        candidate_residues = list(set(seq))
        masses = [constants.AA_MASSES[a].mono for a in candidate_residues]
        candidate_idx = list(range(len(candidate_residues)))
        # combination of 2 residues for insertion
        masses += [m1 + m2 for m1, m2 in itertools.product(masses, masses)]
        candidate_idx += list(itertools.product(candidate_idx, candidate_idx))

        # convert the list to masses
        masses = np.array(masses)

        # partition the sequence in all possibilities
        n = len(seq_x)
        all_seqs_l1 = [tuple([seq_x[:i], seq_x[i:]]) for i in range(n)]
        all_seqs_l2 = [
            tuple([seq_x[i:j] for i, j
                   in zip([0] + list(ix), list(ix) + [None])])
            for ix in itertools.combinations_with_replacement(range(n), 2)
        ]

        # insert residues into the sequence
        new_seqs = []
        for c in range(2, 5):
            mass = mz_neutral * c
            dm = mass - pmass
            # do all possible combinations
            match_ix, = np.where(np.absolute(dm - masses) <= self.tol)
            for i in match_ix:
                ix = candidate_idx[i]
                if isinstance(ix, int):
                    new_seqs += [
                        (c, "".join([s1, candidate_residues[ix], s2]))
                        for s1, s2 in all_seqs_l1
                    ]
                else:
                    new_seqs += [
                        (c, "".join([s1, candidate_residues[ix[0]], s2,
                                     candidate_residues[ix[1]], s3]))
                        for s1, s2, s3 in all_seqs_l2
                    ]

        # parse them back to sequence and modifications
        candidates = []
        for c, pk in new_seqs:
            mods_new = []
            for i, val in mods_bk.items():
                j = pk.index(i)
                mods_new.append(ModSite(val["m"], j+1, val["n"]))
                pk = pk.replace(i, val["r"])
            candidates.append(Peptide(pk, c, pre_mods+mods_new))
        return candidates

    @staticmethod
    def _combine_mods(seq, mods):
        """ Insert modification after target residue. """
        if not mods:
            return seq

        # separate modifications
        term_mods, int_mods = [""] * 2, []
        for mod in mods:
            if isinstance(mod.site, str):
                term_mods[0 if mod.site == "nterm" else 1] = f"[{mod.mod}]"
            else:
                int_mods.append(mod)
        int_mods = sorted(int_mods, key=attrgetter("site"))

        # combine modifications and sequences
        frags, i = [], 0
        for mod in int_mods:
            frags.append(f"{seq[i:mod.site]}[{mod.mod}]")
            i = mod.site
        if i < len(seq):
            frags.append(seq[i:])

        return "".join([term_mods[0], "".join(frags), term_mods[1]])

    @staticmethod
    def _reconstruct_peptide(parsed_match, seq_corr, mods_corr):
        """ Reconstruct the peptides after correction. """
        seq = parsed_match.seq_prefix + seq_corr + parsed_match.seq_suffix
        # re-construct modifications
        mods = parsed_match.mods_prefix.copy()
        npre = len(parsed_match.seq_prefix)
        for mod in mods_corr:
            mods.append(mod if isinstance(mod.site, str) else
                        mod._replace(site=npre+mod.site))
        # end modifications
        npre += len(seq_corr)
        for mod in parsed_match.mods_suffix:
            mods.append(mod._replace(site=npre+mod.site))

        return seq.upper(), mods

    @staticmethod
    def _sequence_tag_search(psm, syn_peptides):
        """ Search sequence tags and match to synthetic peptides. """
        # centroid spectrum
        spectrum = psm.spectrum.centroid()
        # spectrum object
        proc_spectrum = _Spectrum(spectrum._peaks, spectrum.prec_mz)
        # denoise
        _ = proc_spectrum.denoise()
        # get sequence tags
        proc_spectrum.extract_sequence_tags()
        # if no tag is found, which means that the spectrum is bad, return None
        if not proc_spectrum.sequence_tags:
            return None

        # search the peptide using the tags
        matches = []
        for seq, mods in syn_peptides:
            prec_tags = proc_spectrum.match_sequence(seq, mods=mods)
            for tag in prec_tags:
                n = tag.length + 1
                ion_index = dict(
                    zip(("y", "b") if tag.ion_type == "y" else ("b", "y"),
                        (0, n))
                )
                # precursor mass of subsequence containing the tag
                ms = _peptide_mass(tag.seq_term, tag.mods_tag)
                mp = tag.mz_tag - constants.FIXED_MASSES["H"]
                # match object to tag precursor
                m = SynMatch(seq=tag.seq_term, mods=tag.mods_tag, charge=1,
                             num_ions=n, max_tag=n, ion_index=ion_index,
                             prec_mz=tag.mz_tag, delta_mass=[(ms - mp, 1)],
                             nterm_left=False, cterm_left=False)
                matches.append((m, tag))

        return matches

    @staticmethod
    def _tag_match_correct(correct_pep, tag):
        """ Correct the first match to full peptide sequence. """
        seq, mods = tag.pseq, tag.pmods
        n = len(correct_pep.seq)
        # correct precursor sequence using the corrected tag precursor
        if tag.ion_type == "y":
            corr_seq = seq[:tag.index] + correct_pep.seq
            ion_index = {"y": n, "b": 0}
            corr_mods = [mod for mod in mods if isinstance(mod.site, str)
                         or mod.site <= tag.index]
            for mod in correct_pep.mods:
                corr_mods.append(mod._replace(site=mod.site + int(tag.index)))
        else:
            corr_seq = correct_pep.seq + seq[tag.index+1:]
            ion_index = {"b": n, "y": 0}
            corr_mods = list(correct_pep.mods)
            for mod in mods:
                if isinstance(mod.site, int) and mod.site >= tag.index:
                    corr_mods.append(
                        mod._replace(site=mod.site - int(tag.index) + n)
                    )
                elif mod.site == "cterm":
                    corr_mods.append(mod)
        m = _peptide_mass(corr_seq, corr_mods)

        return SynMatch(seq=corr_seq, mods=corr_mods, max_tag=tag.length+1,
                        ion_index=ion_index, delta_mass=m)

    def _validate_candidates(self, psm, candidates):
        """ Validate candidates. """
        spectrum = psm.spectrum.centroid()
        # denoising, including deisotoping
        proc_spectrum = _Spectrum(spectrum._peaks, spectrum.prec_mz)
        denoised_spectrum = proc_spectrum.denoise()
        # prefilter candidates to get top 1000 matches
        candidates = self._prefiltering(candidates, denoised_spectrum)
        # new psms
        x = []
        for pep in candidates:
            _psm_c = PSM(psm.data_id, psm.spec_id, pep, spectrum)
            # get features
            features = _psm_c.extract_features()
            x.append([getattr(features, name) for name in Feature_names])
        # validation scores
        scores = self.validator.decision_function(np.array(x))

        # best match
        j = np.argmax(scores)
        best_psm = PSM(psm.data_id, psm.spec_id, candidates[j], spectrum)
        best_psm.site_score = scores[j]
        _ = best_psm.extract_features()

        return best_psm

    @staticmethod
    def _prefiltering(candidates, spectrum, tol=0.2):
        """
        Filter candidates prior to validation. The best 1000 candidates
        will be returned for further validation.
        """
        if len(candidates) <= 1000:
            return candidates

        aamass = constants.AA_MASSES

        mz, n = np.sort(spectrum[:, 0]), spectrum.shape[0]
        mh, mh2o = constants.FIXED_MASSES["H"], constants.FIXED_MASSES["H2O"]
        # quick annotations simply using b and y ions.
        num_ions = []
        for candidate in candidates:
            # get residue masses
            seq_mass = np.array([aamass[a].mono for a in candidate.seq])
            # singly charged terminal adducts in [N-terminus, C-terminus]
            term_mass = [mh, mh2o + mh]
            for mod in candidate.mods:
                if isinstance(mod.site, int):
                    seq_mass[mod.site - 1] += mod.mass
                else:
                    term_mass[0 if mod.site == "nterm" else 1] += mod.mass
            # singly charged y and b ion m/z
            ybs = np.concatenate(
                (np.cumsum(seq_mass[:-1]) + term_mass[0],
                 np.cumsum(seq_mass[::-1][:-1]) + term_mass[1]),
                axis=0
            )
            ybs.sort()

            # do quick annotation
            mix = [bisect_left(mz, m) for m in ybs]
            nk = sum((k > 0 and m-mz[k-1] <= tol) or (k < n and mz[k]-m <= tol)
                     for k, m in zip(mix, ybs))
            num_ions.append(nk)

        # retain candidates with 1000 highest number of y and b ions annotated.
        sorted_ix = np.argsort(num_ions)[::-1]

        return [candidates[i] for i in sorted_ix[:1000]]

    def _restrict_refs(self, seq: str, mods: typing.List[ModSite],
                       nterm: bool = False, cterm: bool = False):
        """
        Restrict residue combinations and artifacts according to seq.

        Args:
            seq: The sequence for restricting the residue combinations.
            mods: Modifications.

        Returns:
            combs: Restricted combinations of residues.
            combs_add: Restricted combinations of residues as additional
                       residues in permutations of synthesis error.
            artifacts: Artifacts.
            seq_arr: Sequence array.

        """
        nseq = len(seq)
        # get residue mass w/o modifications
        seq_list, mod_dict = list(seq), {}
        seq_mass = np.array([constants.AA_MASSES[a].mono for a in seq])
        seq_mass_mod = seq_mass.copy()

        for i, mod in enumerate(mods):
            if isinstance(mod.site, int):
                mod_dict[f"{i}"] = {
                    "mod": mod.mod, "res": seq[mod.site-1], "mass": mod.mass
                }
                seq_list[mod.site - 1] = f"{i}"
                seq_mass_mod[mod.site - 1] += mod.mass

        # combinations for removing residues, modifications are considered
        combs = collections.defaultdict(dict)
        for i in range(6):
            rm_ix, mass = [], []
            for ix in itertools.combinations(range(nseq), i+1):
                ix2 = list(ix)
                rm_ix.append(ix2)
                mass.append(seq_mass_mod[ix2].sum())
            combs[i+1]["rm_index"] = rm_ix
            combs[i+1]["mass"] = np.array(mass)

        # combinations for adding residues, modifications are excluded
        res = ["".join(seq[j] for j in ix) for i in range(2)
               for ix in itertools.combinations(range(nseq), i+1)]
        mass = np.array([seq_mass[list(ix)].sum() for i in range(2)
                         for ix in itertools.combinations(range(nseq), i+1)])
        combs_add = {"residues": res, "mass": mass}

        # consider artifact modifications
        seq_n = list(seq_list)
        if nterm:
            seq_n.append("nterm")
        if cterm:
            seq_n.append("cterm")
        res, mass = [], []
        for r in seq_n:
            if r in self.artifacts:
                res += [[mod, r] for mod, _ in self.artifacts[r]]
                mass += [m for _, m in self.artifacts[r]]
        artifacts = {"mods": np.array(res, dtype=str), "mass": np.array(mass)}

        return combs, combs_add, artifacts, seq_list, mod_dict

    @staticmethod
    def _parse_matches(match):
        """
        Parse sequence into three regions according to tags:
            seq_prefix: first part
            seq_proc: middle part, which will be the target for error
                      correction.
            seq_suffix: the final part.
        """
        # mass of matched synthetic peptide
        seq, mods = match.seq, match.mods
        mass = [constants.AA_MASSES[a].mono for a in seq]

        # indices define subsequence for analysis
        nq = len(match.seq)
        if match.ion_index is not None:
            b, y = match.ion_index["b"], match.ion_index["y"]
            j0, j1 = (b, nq - y) if nq - y >= b else (nq - y, b)
            j0, j1 = max(j0 - 1, 0), min(nq, j1 + 2)
        else:
            j0, j1 = 0, nq

        # sequence and modifications for processing
        s0, s1, s2 = seq[:j0], seq[j0:j1], seq[j1:]
        m0, _, m2 = sum(mass[:j0]), sum(mass[j0:j1]), sum(mass[j1:])
        # separate modifications
        mod0, mod1, mod2 = [], [], []
        for mod in mods:
            if isinstance(mod.site, str) or mod.site <= j0:
                mod0.append(mod)
                m0 += mod.mass
            elif mod.site > j1:
                mod2.append(mod._replace(site=int(mod.site-j1)))
                m2 += mod.mass
            else:
                mod1.append(mod._replace(site=int(mod.site-j0)))

        return ProcMatch(seq_prefix=s0, mass_prefix=m0, mods_prefix=mod0,
                         seq_proc=s1, mods_proc=mod1, seq_suffix=s2,
                         mass_suffix=m2, mods_suffix=mod2)

    @staticmethod
    def _parse_raw_id(raw):
        """ Parse raw id. """
        raw_split = raw.split("_")
        j = [i for i, x in enumerate(raw_split) if x.endswith("mod")][0]
        target = "_".join(raw_split[j:j+2])
        if target in target_pairs:
            return target_pairs[target]
        return target

    def _get_targets(self):
        """ Get target information. """
        self.targets = json.load(open(r"ptm_experiment_info.json", "r"))

    def _get_synthetic_peptides(self):
        """ Load synthetic peptides. """
        # constants
        mo, mc = 15.994915, 57.021464
        # load synthetic peptides
        syn_peps = collections.defaultdict()
        # load unmodified peptides
        for target in self.targets.keys():
            name = self.targets[target]["unimod_name"]
            m = self.targets[target]["mass"]
            file_ = os.path.join(self.targets[target]["benchmark_path"],
                                 self.targets[target]["benchmark"])
            # load peptides
            mpeps = open(file_, "r").read().splitlines()
            benchmarks = []
            for pep in mpeps:
                # parse peptide to get modifications and sequences
                pep_sep = re.split("\[|\]", pep)
                seq, mods = "", []
                for i in range(1, len(pep_sep), 2):
                    seq += pep_sep[i - 1]
                    mods.append(ModSite(m, len(seq) if seq else "nterm", name))
                if i < len(pep_sep) - 1 or len(pep_sep) == 1:
                    seq += pep_sep[-1]

                # fixed modification at Cysteine
                if "C" in seq:
                    mods += [ModSite(mc, i + 1, "Carbamidomethyl")
                             for i, r in enumerate(seq) if r == "C"]
                    mods.sort(key=attrgetter("site"))
                benchmarks.append((seq, mods))

                # variable modification at Methionine
                _mets = [i + 1 for i, r in enumerate(seq) if r == "M"]
                for j in range(1, len(_mets) + 1):
                    for _ix in itertools.combinations(_mets, j):
                        mvar = [ModSite(mo, i, "Oxidation") for i in _ix]
                        benchmarks.append(
                            (seq, sorted(mods+mvar, key=attrgetter("site")))
                        )
            syn_peps[target] = benchmarks
        self.syn_peps = syn_peps

    def _get_artifact(self):
        """
        Get artifacts.

        Returns:
            mods: numpy array of modifications with sites
            mass: numpy array of mass of residue combinations

        """
        # artifacts
        with open(r"artifiacts.json", "r") as f:
            artifacts = json.load(f)

        # parse the modifications
        res = collections.defaultdict(list)
        for mod, vals in artifacts.items():
            for site in vals["sites"]:
                res[site].append((mod, vals["mass"]))
        self.artifacts = res
