from fairseq.data import Dictionary


class MaskedDictionary(Dictionary):
    def __init__(
        self, pad="[PAD]", eos="</s>", bos="<s>", unk="[UNK]", mask="[MASK]",
    ) -> None:
        super().__init__(pad, eos, unk, bos, extra_special_symbols=["<nl>"])
        self.mask_word = mask
        self.mask_index = self.add_symbol(mask)
        self.nspecial = 99

    def mask(self) -> int:
        return self.mask_index

    @classmethod
    def load(cls, filepath: str) -> "MaskedDictionary":
        dictionary = cls()

        with open(
            filepath, mode="r", encoding="utf-8", errors="ignore"
        ) as input_file:
            for line in input_file:
                key, _ = line.split(" ")
                dictionary.add_symbol(key)

        dictionary.nspecial = 99

        return dictionary
