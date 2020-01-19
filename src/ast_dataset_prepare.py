import os
import typing
from pathlib import Path
from .ast_parser import ASTParser, LanguageRepr


class ASTParseRule(Enum):
    root_nodes = 1
    all_nodes = 2


    def __init__(self, parser: ASTParser, lang_ext: str):
class ASTFileParser:
        self.parser = parser
        self.__lang_ext = lang_ext

    def dump_files(
        self,
        repository_path: str,
        output_path: str,
        exts: typing.Tuple[str, str],
    ) -> None:
        # dirname = os.path.dirname(output_path)
        os.makedirs(output_path, exist_ok=True)
        repository_path = Path(repository_path)

        for path in repository_path.glob(f"**/*.{self.__lang_ext}"):
            file_path = str(path.absolute())
            basename = self.__file_basename(file_path)
            input_file = open(file_path, mode="r")

            prefix = output_path.replace("/", "_") + "_"
            source_filepath = os.path.join(
                output_path, prefix + basename + "." + exts[0]
            )
            target_filepath = os.path.join(
                output_path, prefix + basename + "." + exts[1]
            )

            source_file = open(source_filepath, mode="w")
            target_file = open(target_filepath, mode="w")

            with input_file, source_file, target_file:
                program = "".join(input_file.readlines())
                parsed_nodes = self.parser.parse_root_children(program)
                for src, ast in parsed_nodes:
                    source_file.write(src + "\n")
                    target_file.write(ast + "\n")
                # break

    def __file_basename(self, filepath: str) -> str:
        basename = os.path.basename(filepath)
        basename, _ = os.path.splitext(basename)
        return basename


class ASTParserBuilder:
    def __init__(self, library_path: str, language: str):
        self.library_path = library_path
        self.language = language

    def build(self) -> ASTParser:
        lang_repr = LanguageRepr(self.library_path, self.language)
        parser = ASTParser(lang_repr)
        return parser

def main():
    lang_repr = LanguageRepr(
        library_path="/workspace/tmp/ast_test/my-languages.so", lang="python"
    )

    parser = ASTParser(lang_repr)
    dumper = ASTRepoFileDumper(parser, "py")
    dumper.dump_files(
        "/workspace/tmp/ast_test/transformers",
        "/workspace/tmp/ast_test/transformers_output",
        ("src", "ast"),
    )


if __name__ == "__main__":
    main()
