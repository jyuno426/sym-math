# -*- coding: utf-8 -*-

__all__ = ["Tree", "Node"]


class Tree(object):
    def __init__(self, root=None):
        self.root = root
        if self.root is not None:
            assert type(root) == Node

    def get_expr(self, init=False):
        return self.root.get_expr(init)

    def get_leaf_list(self):
        return [node for node in self.root.traverse_in_preorder() if node.is_leaf()]


class Node(object):
    def __init__(self, data=None):
        self.data = data
        # self.parent = None
        self.children = []

        # variables should be sync always:
        # self.symbols = set()
        self.expr = None

    def is_leaf(self):
        return len(self.children) == 0

    def is_unary(self):
        return len(self.children) == 1

    def is_binary(self):
        return len(self.children) == 2

    def add_child(self, child):
        assert type(child) == Node
        self.children.append(child)
        # child.parent = self
        return child

    def get_expr(self, init=False):
        if self.expr is None or init:
            if self.is_leaf():
                self.expr = self.data
            else:
                self.expr = self.data(
                    *[child.get_expr(init) for child in self.children]
                )
        return self.expr

    def traverse_in_preorder(self):
        traverse = [self]
        for child in self.children:
            traverse += child.traverse_in_preorder()
        return traverse
