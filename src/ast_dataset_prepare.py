import os
import typing
from enum import Enum
from pathlib import Path
from .ast_parser import ASTParser, LanguageRepr


class ASTParseRule(Enum):
    root_nodes = 1
    all_nodes = 2


class ASTFileParser:
    def __init__(self, parser: ASTParser, rule: ASTParseRule):
        self.parser = parser
        self.rule = rule

    def parse_file(
        self,
        filepath: str,
        output_path: str,
        extensions: typing.Tuple[str, str],
    ) -> None:
        os.makedirs(output_path, exist_ok=True)
        basename = self.__file_basename(filepath)
        prefix = os.path.dirname(filepath).replace("/", "_") + "_"

        source_filepath = os.path.join(
            output_path, prefix + basename + "." + extensions[0]
        )
        target_filepath = os.path.join(
            output_path, prefix + basename + "." + extensions[1]
        )

        input_file = open(filepath, mode="r")
        source_file = open(source_filepath, mode="w")
        target_file = open(target_filepath, mode="w")

        with input_file, source_file, target_file:
            program = "".join(input_file.readlines())
            if self.rule == ASTParseRule.all_nodes:
                parsed_nodes = self.parser.parse_program(program)
            else:
                parsed_nodes = self.parser.parse_root_children(program)

            for src, ast in parsed_nodes:
                source_file.write(src + "\n")
                target_file.write(ast + "\n")

    @staticmethod
    def __file_basename(filepath: str) -> str:
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
