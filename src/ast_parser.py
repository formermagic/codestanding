import os
import shutil
import typing
from collections import deque
from typing import List, Optional, Tuple

from git import Repo
from tree_sitter import Language, Node, Parser, TreeCursor


class TreeNode:
    def __init__(
        self,
        node: Node,
        start_point: Optional[Tuple[int, int]] = None,
        end_point: Optional[Tuple[int, int]] = None,
    ):
        self.node: Node = node
        self.children: List[Node] = node.children
        self.__start_point = start_point or node.start_point
        self.__end_point = end_point or node.end_point

    def sexp(self) -> str:
        return self.node.sexp()

    @property
    def start_point(self) -> Tuple[int, int]:
        return self.__start_point

    @property
    def end_point(self) -> Tuple[int, int]:
        return self.__end_point

    @property
    def type(self) -> str:
        return self.node.type

    @property
    def is_named(self) -> bool:
        return self.node.is_named


class LanguageRepr:
    def __init__(self, library_path: str, lang: str):
        self.__library_path = library_path
        self.__lang = lang
        self.language = self.__built_language()

    def __built_language(self) -> Language:
        return Language(self.__library_path, self.__lang)

    @property
    def parser(self) -> Parser:
        _parser = Parser()
        _parser.set_language(self.language)
        return _parser


class LanguageReprBuilder:
    def __init__(self, repo_path: str, output_path: str):
        self.repo_path = repo_path
        self.output_path = output_path

    def __clone_repo(self) -> str:
        url = "https://github.com/tree-sitter/tree-sitter-python"
        output_path = os.path.join(self.repo_path, "tree-sitter-python")

        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.makedirs(self.repo_path, exist_ok=True)
        _ = Repo.clone_from(url, output_path)

        return output_path

    def build(self, lang: str, removing_tmp: bool = True) -> LanguageRepr:
        repo_path = self.__clone_repo()
        Language.build_library(self.output_path, repo_paths=[repo_path])
        if removing_tmp:
            shutil.rmtree(repo_path)
        lang_repr = LanguageRepr(library_path=self.output_path, lang=lang)
        return lang_repr


class ASTParser:
    def __init__(self, language_repr: LanguageRepr):
        self.__language_repr = language_repr
        self.parser = language_repr.parser

    def parse_program(
        self, program: str
    ) -> typing.List[typing.Tuple[str, str]]:
        tree = self.parser.parse(bytes(program, "utf8"))
        root_node = TreeNode(tree.root_node)

        program_lines = program.split("\n")
        src_lines: typing.List[str] = []
        ast_lines: typing.List[str] = []

        for node in self.traverse_tree(root_node):
            if (
                node.type == "class_definition"
                or node.type == "function_definition"
            ):
                src, ast = self.__parse_def(node, program_lines)
                src_lines.append(src)
                ast_lines.append(ast)
            elif node.type == "decorated_definition":
                src, ast = self.__parse_decorated_def(node, program_lines)
                src_lines.append(src)
                ast_lines.append(ast)

            if "statement" in node.type or "definition" in node.type:
                src_lines.append(self.find_substring(program_lines, node))
                ast_lines.append(self.parse_node(node, program_lines))
                # print(f"NodeType: {node.type}")

        return list(zip(src_lines, ast_lines))

    def parse_root_children(
        self, program: str
    ) -> typing.List[typing.Tuple[str, str]]:
        tree = self.parser.parse(bytes(program, "utf8"))
        program_lines = program.split("\n")

        raw_bodies: typing.List[str] = []
        parsed_children: typing.List[str] = []

        children = deque([TreeNode(node) for node in tree.root_node.children])

        while children:
            root_child = children.popleft()
            if root_child.type == "class_definition":
                block_node = root_child.children[-1]
                for node in block_node.children[::-1]:
                    children.appendleft(TreeNode(node))

                source_code, parsed_ast = self.__parse_def(
                    root_child, program_lines
                )

                raw_bodies.append(source_code)
                parsed_children.append(parsed_ast)
            elif root_child.type == "function_definition":
                source_code, parsed_ast = self.__parse_def(
                    root_child, program_lines
                )

                raw_bodies.append(source_code)
                parsed_children.append(parsed_ast)
            elif root_child.type == "decorated_definition":
                for node in root_child.children[::-1]:
                    children.appendleft(TreeNode(node))

            raw_bodies.append(self.find_substring(program_lines, root_child))
            parsed_children.append(self.parse_node(root_child, program_lines))

        return list(zip(raw_bodies, parsed_children))

    def __parse_def(
        self, node: TreeNode, program_lines: List[str]
    ) -> Tuple[str, str]:
        start_point = node.children[0].start_point
        end_point = node.children[0].end_point
        for child in node.children:
            if child.type == "block":
                break
            end_point = child.end_point

        definition_node = TreeNode(node.node, start_point, end_point)
        source_code = self.find_substring(program_lines, definition_node)
        parsed_ast = self.parse_node(definition_node, program_lines)

        # drop body from node definition
        parsed_ast = parsed_ast.split(" body:")[0] + ")"

        return source_code, parsed_ast

    def __parse_decorated_def(
        self, node: TreeNode, program_lines: List[str]
    ) -> Tuple[str, str]:
        start_point = node.children[0].start_point
        def_nodes = node.children[1].children
        end_point = def_nodes[0].end_point
        for child in def_nodes:
            if child.type == "block":
                break
            end_point = child.end_point

        definition_node = TreeNode(node.node, start_point, end_point)
        src = self.find_substring(program_lines, definition_node)
        ast = self.parse_node(definition_node, program_lines)

        # drop body from node definition
        ast = ast.split(" body:")[0] + ")"

        return src, ast

    def parse_node(self, node: TreeNode, program_lines: List[str]) -> str:
        source_sexp = node.sexp()
        for child_node in self.traverse_tree(node):
            if child_node.type == "identifier":
                identifier = self.find_substring(program_lines, child_node)
                source_sexp = source_sexp.replace("identifier", identifier, 1)
        return source_sexp

    # def parse_program(self, program: str) -> str:
    #     tree = self.parser.parse(bytes(program, "utf8"))
    #     source_sexp = tree.root_node.sexp()
    #     program_lines = program.split("\n")

    #     for node in self.traverse_tree(tree.root_node):
    #         if node.type == "identifier":
    #             identifier = self.find_substring(program_lines, node)
    #             source_sexp = source_sexp.replace("identifier", identifier, 1)

    #     return source_sexp

    def find_substring(self, program_lines: List[str], node: TreeNode) -> str:
        start_point, end_point = node.start_point, node.end_point
        lines: typing.List[str] = []

        for idx in range(start_point[0], end_point[0] + 1):
            start_idx, end_idx = 0, len(program_lines[idx])
            if idx == start_point[0]:
                start_idx = start_point[1]
            if idx == end_point[0]:
                end_idx = end_point[1]
            lines.append(program_lines[idx][start_idx:end_idx])

        return "<nl>".join(lines)

    def traverse_tree(self, node: TreeNode) -> TreeNode:
        node_deque = deque(node.children)
        while node_deque:
            left_node = TreeNode(node_deque.popleft())
            if left_node.is_named:
                node_deque.extendleft(left_node.children[::-1])
                yield left_node


def test_language_building():
    builder = LanguageReprBuilder(
        repo_path="/workspace/tmp/ast_test",
        output_path="/workspace/tmp/ast_test/my-languages.so",
    )

    language = builder.build("python")


def test_language_loading():
    language = LanguageRepr(
        library_path="/workspace/tmp/ast_test/my-languages.so", lang="python"
    )

    print("Loaded a language!")


def main() -> None:
    lang_repr = LanguageRepr(
        library_path="/workspace/tmp/ast_test/my-languages.so", lang="python"
    )

    parser = ASTParser(lang_repr)

    program = """\
    import os
    from os.path import join

    def split_path(path: str) -> str:
        base = os.path.base(path)
        joined_path = join(base, "123.json")
        return joined_path
        
    def say_hello(name: str) -> None:
        print(f"Hello, {name}!")\
    """

    parsed = parser.parse_root_children(
        program
    )  # parser.parse_program(program)

    for _code, _ast in parsed:
        print(_code)
        print(_ast)
        print()


if __name__ == "__main__":
    main()
