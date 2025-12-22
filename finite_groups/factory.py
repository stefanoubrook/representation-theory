import re

import numpy as np

from .group import FiniteGroup


class GroupFactory:
    @staticmethod
    def cyclic_group(n: int) -> FiniteGroup:
        # Generates the Cyclic group C_n
        elements = [f"z{i}" for i in range(n)]

        range_arr = np.arange(n)
        table = (range_arr + range_arr[:, None]) % n

        return FiniteGroup(elements, table)

    @staticmethod
    def parse_presentation(presentation_str: str):
        clean_str = presentation_str.strip("<> ")

        if "|" not in clean_str:
            raise ValueError(
                "Presentation must contain '|' to seperate genertors from relations"
            )

        gen_part, rel_part = clean_str.split("|")

        generators = [g.strip() for g in gen_part.split(",") if g.strip()]

        relations = [r.strip() for r in rel_part.split(",") if r.strip()]

        return generators, relations

    @staticmethod
    def expand_relations(rel: str):
        pattern = r"\((.*?)\)\^(\d+)|(\w+)\^(\d+)"

        def replacer(match):
            base = match.group(1) or match.group(3)
            exponent = match.group(2) or match.group(4)

            return base * int(exponent)

        while "^" in rel:
            rel = re.sub(pattern, replacer, rel)
        return rel

    @staticmethod
    def simplify_word(word: str, rules: list[tuple[str, str]]) -> str:
        # Repeatedly applies reduction rules to a word until no more rules can be applied
        changed = True
        while changed:
            changed = False
            for lhs, rhs in rules:
                if lhs in word:
                    word = word.replace(lhs, rhs, 1)
                    changed = True
                    break
        return word

    @staticmethod
    def from_presentation(presentation_str: str):
        generators, raw_relations = GroupFactory.parse_presentation(presentation_str)
        # Expand + normalize rules
        rules = []
        for r in raw_relations:
            expanded = GroupFactory.expand_relations(r)
            if "=" in expanded:
                lhs, rhs = expanded.split("=")
                rhs = "" if rhs.strip() in ["e", "1"] else rhs.strip()
                rules.append((lhs.strip(), rhs))
            else:
                rules.append((expanded.strip(), ""))

        # BFS to find elements
        elements = [""]
        queue = [""]

        while queue:
            current_word = queue.pop(0)
            for gen in generators:
                new_word = current_word + gen

                # Apply group relations
                reduced_word = GroupFactory.simplify_word(new_word, rules)

                if reduced_word not in elements:
                    elements.append(reduced_word)
                    queue.append(reduced_word)

        n = len(elements)
        table = np.zeros((n, n), dtype=int)
        element_to_ind = {word: i for i, word in enumerate(elements)}

        for i, row_word in enumerate(elements):
            for j, col_word in enumerate(elements):
                combined = row_word + col_word
                simplified = GroupFactory.simplify_word(combined, rules)
                table[i, j] = element_to_ind[simplified]
        return FiniteGroup(elements, table)
