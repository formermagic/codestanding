"""Parse ASTs for all source code files in the given directory.
Produce a pair of files (code, ast) for every source code file.

Usage:
    ast_dataset_prepare parse-nodes \
        [--rule-all | --rule-root] \
        --library-path=<lib> \
        --language=<lang> \
        --language-ext=<lang_ext> \
        --root-input-path=<inp> \
        --output-path=<out> \
        --extensions=<exts>

Options:
    --rule-all                  A rule to parse all statements in source files.
    --rule-root                 A rule to parse only root nodes in source files.
    --library-path=<lib>        A path to the built tree-sitter library for supported lanuages.
    --language=<lang>           A language to parse.
    --language-ext=<lang_ext>   A language extension to look for while walking through files.
    --root-input-path=<inp>     An input directory to walk through looking for files to parse.
    --output-path=<out>         An output directory to write parsed pairs (code, ast) to.
    --extensions=<exts>         Extensions for parsed pair files (code, ast).
"""
import os
import typing
from enum import Enum
from pathlib import Path
from .ast_parser import ASTParser, LanguageRepr
from .workable import Workable, WorkableRunner


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


class ASTFileParserWorkable(Workable):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        parser_builder: ASTParserBuilder,
        parser_rule: ASTParseRule,
        filepath: Path,
        output_path: str,
        extensions: typing.Tuple[str, str],
    ) -> None:
        self.parser_builder = parser_builder
        self.parser_rule = parser_rule
        self.filepath = filepath
        self.output_path = output_path
        self.extensions = extensions

    def run(self) -> None:
        filepath = str(self.filepath)
        parser = self.parser_builder.build()
        file_parser = ASTFileParser(parser, self.parser_rule)
        file_parser.parse_file(filepath, self.output_path, self.extensions)


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
