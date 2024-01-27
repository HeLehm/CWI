from simplify_zeroshot import paraphrase_beam_search
import re
from readability import Readability
from tqdm import tqdm
def remove_ansi_codes(text):
    """
    Remove ANSI escape codes from a string.
    """
    ansi_escape = re.compile(r'''
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]
        |     # or [ for CSI, followed by a control sequence
            \[
            [0-?]*  # Parameter bytes
            [ -/]*  # Intermediate bytes
            [@-~]   # Final byte
        )
    ''', re.VERBOSE)
    return ansi_escape.sub('', text)


if __name__=="__main__":
    #read in dataset
    ds = ["The proliferation of technologically advanced gadgets has substantially augmented the efficacy of our daily communications."]
    with open("asset.test.orig", "r")as f:
        ds = f.readlines()
    # a lot of annoying saving but it works
    original = ""
    original_save = []
    without_complete = ""
    without_complete_save = []
    with_complete = ""
    with_complete_save = []
    for x,sample in tqdm(enumerate(ds)):
        if x > 100:
            break
        #without CWI:
        print(sample)
        #both return options - take the first one for without CWI and the best one (accoridng to CWI) for with CWI
        without_cwi, loss = paraphrase_beam_search(sample, cwi=False)
        without_cwi = remove_ansi_codes(without_cwi[0])

        
        with_cwi,loss = paraphrase_beam_search(sample, cwi=True)
        min_value = min(loss)
        min_index = loss.index(min_value)
        with_cwi = remove_ansi_codes(with_cwi[min_index])
        
        print(f"without CWI: {without_cwi}")
        print(f"with CWI: {with_cwi}")

        original += " "+sample
        original_save.append(sample)
        without_complete += " "+without_cwi
        without_complete_save.append(without_cwi)
        with_complete += " "+ with_cwi
        with_complete_save.append(with_cwi)

    with open("original.txt", "w") as f:
        for string in original_save:
            f.write(string + "\n")
    with open("without_complete.txt", "w") as f:
        for string in without_complete_save:
            f.write(string + "\n")
    with open("with_complete.txt", "w") as f:
        for string in with_complete_save:
            f.write(string + "\n")

    print(f"original: {Readability(original).flesch()}")
    print(f"without CWI score: {Readability(without_complete).flesch()}")
    print(f"with CWI score: {Readability(with_complete).flesch()}")
