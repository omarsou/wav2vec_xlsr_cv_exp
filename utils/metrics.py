from datasets import load_metric
wer_metric = load_metric("wer")


### PER ###
def NeedlemanWunschAlignScore(seq1, seq2, d, m, r, normalize=True):
    N1, N2 = len(seq1), len(seq2)

    # Fill up the errors
    tmpRes_ = [[None for x in range(N2 + 1)] for y in range(N1 + 1)]
    for i in range(N1 + 1):
        tmpRes_[i][0] = i * d
    for j in range(N2 + 1):
        tmpRes_[0][j] = j * d

    for i in range(N1):
        for j in range(N2):
            match = r if seq1[i] == seq2[j] else m
            v1 = tmpRes_[i][j] + match
            v2 = tmpRes_[i + 1][j] + d
            v3 = tmpRes_[i][j + 1] + d
            tmpRes_[i + 1][j + 1] = max(v1, max(v2, v3))

    i = j = 0
    res = -tmpRes_[N1][N2]
    if normalize:
        res /= float(N1)
    return res


def get_seq_PER(seqLabels, detectedLabels):
    return NeedlemanWunschAlignScore(seqLabels, detectedLabels, -1, -1, 0,
                                     normalize=True)


def generate_per_score(refs, hyps):
    score = 0.0
    for ref, hyp in zip(refs, hyps):
        score += get_seq_PER(ref.replace('[UNK]', ''), hyp.replace('[UNK]', ''))
    return score / len(refs)


### WER ###
def compute_wer(pred, ref):
    wer = wer_metric.compute(predictions=pred, references=ref)
    return {"wer": wer}
