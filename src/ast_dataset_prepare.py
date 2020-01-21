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
import logging
import os
import subprocess
import typing
import uuid
from enum import Enum
from pathlib import Path

from docopt import docopt

from .ast_parser import ASTParser, LanguageRepr
from .utils import parse_listed_arg
from .workable import Workable, WorkableRunner


class ASTParseRule(Enum):
    root_nodes = 1
    all_nodes = 2


class ASTFileParser:
    def __init__(self, parser: ASTParser, rule: ASTParseRule):
        self.parser = parser
        self.rule = rule

    # pylint: disable=broad-except
    def parse_file(
        self,
        filepath: str,
        output_path: str,
        extensions: typing.Tuple[str, str],
    ) -> None:
        os.makedirs(output_path, exist_ok=True)
        # basename = self.__file_basename(filepath)
        # prefix = os.path.dirname(filepath).replace("/", "_") + "_"
        prefix = str(uuid.uuid4())

        source_filepath = os.path.join(
            output_path, prefix + "." + extensions[0]
        )
        target_filepath = os.path.join(
            output_path, prefix + "." + extensions[1]
        )

        logging.info("Parsing file %s", filepath)

        try:
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
        except OSError as error:
            logging.error(
                "OSError ocurred while parsing file: %s, exception: %s",
                filepath,
                str(error),
            )
        except UnicodeError as error:
            logging.error(
                "UnicodeError ocurred while parsing file: %s, exception: %s",
                filepath,
                str(error),
            )
        except BaseException as error:
            logging.error(
                "Exception ocurred while parsing file: %s, exception: %s",
                filepath,
                str(error),
            )

        logging.info("Finished parsing file %s", filepath)

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

def find_source_files(
    root_input_path: str, language_ext: str, files_path: str
) -> None:
    with open(files_path, mode="w") as output:
        subprocess.run(
            ["find", root_input_path, "-name", f"*.{language_ext}", "-print"],
            check=True,
            stdout=output,
        )




def main():
    """
    Usage:
        python -m src.ast_dataset_prepare parse-nodes --rule-all \
            --library-path=/workspace/tmp/code2ast_large/langs.so \
            --language=python \
            --language-ext=py \
            --root-input-path=/workspace/tmp/code2ast_large/repositories \
            --output-path=/workspace/tmp/code2ast_large/_parsed_files \
            --extensions="src, ast"
    """

    # parse arguments
    arguments = docopt(__doc__, version="Remove duplicates 1.0")

    library_path = str(arguments["--library-path"])
    language = str(arguments["--language"])
    language_ext = str(arguments["--language-ext"])

    root_input_path = str(arguments["--root-input-path"])
    output_path = str(arguments["--output-path"])
    extensions: typing.Tuple[str, str] = parse_listed_arg(
        arguments["--extensions"]
    )

    # prepare workables to run
    if arguments["--rule-root"]:
        parse_rule = ASTParseRule.root_nodes
    elif arguments["--rule-all"]:
        parse_rule = ASTParseRule.all_nodes
    else:
        parse_rule = ASTParseRule.all_nodes

    parse_builder = ASTParserBuilder(library_path, language)
    filepaths = Path(root_input_path).glob(f"**/*.{language_ext}")

    workables = [
        ASTFileParserWorkable(
            parse_builder, parse_rule, path, output_path, extensions
        )
        for path in filepaths
    ]

    # run workables
    runner = WorkableRunner()
    runner.execute(workables, max_workers=16)


if __name__ == "__main__":
    main()
