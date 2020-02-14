#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 3
parser_transitions.py: Algorithms for completing partial parsess.
Sahil Chopra <schopra8@stanford.edu>
"""

import sys


class PartialParse(object):
    def __init__(self, sentence):
        """Initializes this partial parse.

        @param sentence (list of str): The sentence to be parsed as a list of words.
                                        Your code should not modify the sentence.
        """
        # The sentence being parsed is kept for bookkeeping purposes. Do not alter it in your code.
        self.sentence = sentence

        # YOUR CODE HERE (3 Lines)
        # Your code should initialize the following fields:
        # self.stack: The current stack represented as a list with the top of the stack as the
        # last element of the list.
        # self.buffer: The current buffer represented as a list with the first item on the
        # buffer as the first item of the list
        # self.dependencies: The list of dependencies produced so far. Represented as a list of
        # tuples where each tuple is of the form (head, dependent).
        # Order for this list doesn't matter.
        ###
        # Note: The root token should be represented with the string "ROOT"
        ###

        self.stack = ["ROOT"]
        self.buffer = sentence.copy()
        self.dependencies = []

        # END YOUR CODE

    def parse_step(self, transition):
        """Performs a single parse step by applying the given transition to this partial parse

        @param transition (str): A string that equals "S", "LA", or "RA" representing the shift,
                                left-arc, and right-arc transitions. You can assume the provided
                                transition is a legal transition.
        """
        # YOUR CODE HERE (~7-10 Lines)
        # TODO:
        # Implement a single parsing step, i.e. the logic for the following as
        # described in the pdf handout:
        # 1. Shift
        # 2. Left Arc
        # 3. Right Arc
        
        ####################################################################################################################################

        if transition == "S":
            if self.buffer:  # ensure buffer is not empty before we perform the transition
                self.stack.append(self.buffer[0])
                self.buffer.pop(0)
        elif transition == "LA":
            if not self.stack[-1] == 'ROOT' and len(self.stack) > 1:
                self.dependencies.append((self.stack[-1], self.stack[-2]))
                self.stack.pop(-2)
        elif transition == 'RA':
            if not self.stack[-1] == 'ROOT' and len(self.stack) > 1:
                self.dependencies.append((self.stack[-2], self.stack[-1]))
                self.stack.pop(-1)

        # END YOUR CODE
        ####################################################################################################################################

    def parse(self, transitions):
        """Applies the provided transitions to this PartialParse

        @param transitions (list of str): The list of transitions in the order they should be applied

        @return dsependencies (list of string tuples): The list of dependencies produced when
                                                        parsing the sentence. Represented as a list of
                                                        tuples where each tuple is of the form (head, dependent).
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    @param sentences (list of list of str): A list of sentences to be parsed
                                            (each sentence is a list of words and each word is of type string)
    @param model (ParserModel): The model that makes parsing decisions. It is assumed to have a function
                                model.predict(partial_parses) that takes in a list of PartialParses as input and
                                returns a list of transitions predicted for each parse. That is, after calling
                                    transitions = model.predict(partial_parses)
                                transitions[i] will be the next transition to apply to partial_parses[i].
    @param batch_size (int): The number of PartialParses to include in each minibatch


    @return dependencies (list of dependency lists): A list where each element is the dependencies
                                                    list for a parsed sentence. Ordering should be the
                                                    same as in sentences (i.e., dependencies[i] should
                                                    contain the parse for sentences[i]).
    """
    dependencies = []

    # YOUR CODE HERE (~8-10 Lines)
    # TODO:
    # Implement the minibatch parse algorithm as described in the pdf handout
    ###
    # Note: A shallow copy (as denoted in the PDF) can be made with the "=" sign in python, e.g.
    # unfinished_parses = partial_parses[:].
    # Here `unfinished_parses` is a shallow copy of `partial_parses`.
    # In Python, a shallow copied list like `unfinished_parses` does not contain new instances
    # of the object stored in `partial_parses`. Rather both lists refer to the same objects.
    # In our case, `partial_parses` contains a list of partial parses. `unfinished_parses`
    # contains references to the same objects. Thus, you should NOT use the `del` operator
    # to remove objects from the `unfinished_parses` list. This will free the underlying memory that
    # is being accessed by `partial_parses` and may cause your code to crash.

    ####################################################################################################################################

    # Instruction from pdf handout: Initialize partial_parses as a list of PartialParses, one for each sentence in sentences
    # convert a list of sentences to a list of PartialParse objects
    partial_parse_objs = [PartialParse(sentence) for sentence in sentences]

    # Instruction from pdf handout: Initialize unfinished_parses as a shallow copy of partial parses
    # unfinished_parse_objs will be dynamically changed in the loop, partial_parse_objs will remain unchanged
    unfinished_parse_objs = partial_parse_objs

    # Instruction from pdf handout: while unfinished_parses is not empty do
    while unfinished_parse_objs:

        # Instruction from pdf handout: Take the first batch_size of parses in unfinished_parses as a minibatch
        parse_objs_mini_batch = unfinished_parse_objs[0:batch_size]

        # while parse_objs_mini_batch is not empty do
        while parse_objs_mini_batch:

            # Instruction from pdf handout: Use the model to predict the next transition for each partial_parse in the minibatch
            # next_transitions_mini_batch is a list (next transition for each parse_obj, just one transition step)
            next_transitions_mini_batch = model.predict(parse_objs_mini_batch)

            # Instruction from pdf handout: Perform a parse step on each partial parse in the minibatch with its predicted transition
            # loop through the list of next transition for each parse_obj
            for i in range(len(next_transitions_mini_batch)):
                # perform parse transitions for each parse_obj
                parse_objs_mini_batch[i].parse_step(
                    next_transitions_mini_batch[i])

            # Instruction from pdf handout: Remove the completed (parse_obj with empty buffer and stack of size 1) parses from unfinished_parses
            for parse_obj in parse_objs_mini_batch:  # loop through all objects in the minibatch
                if not parse_obj.buffer and len(parse_obj.stack) == 1:
                    parse_objs_mini_batch.remove(parse_obj)

        # remove the processed minibatch from unfinished_parse_objs
        unfinished_parse_objs = unfinished_parse_objs[batch_size:]

    for parse_obj in partial_parse_objs:
        # add the dependencies of one parse_obj to the list of dependencies
        dependencies.append(parse_obj.dependencies)

    # END YOUR CODE
    ####################################################################################################################################

    return dependencies


def test_step(name, transition, stack, buf, deps,
              ex_stack, ex_buf, ex_deps):
    """Tests that a single parse step returns the expected output"""
    pp = PartialParse([])
    pp.stack, pp.buffer, pp.dependencies = stack, buf, deps

    pp.parse_step(transition)
    stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer),
                        tuple(sorted(pp.dependencies)))
    assert stack == ex_stack, \
        "{:} test resulted in stack {:}, expected {:}".format(
            name, stack, ex_stack)
    assert buf == ex_buf, \
        "{:} test resulted in buffer {:}, expected {:}".format(
            name, buf, ex_buf)
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(
            name, deps, ex_deps)
    print("{:} test passed!".format(name))


def test_parse_step():
    """Simple tests for the PartialParse.parse_step function
    Warning: these are not exhaustive
    """
    test_step("SHIFT", "S", ["ROOT", "the"], ["cat", "sat"], [],
              ("ROOT", "the", "cat"), ("sat",), ())
    test_step("LEFT-ARC", "LA", ["ROOT", "the", "cat"], ["sat"], [],
              ("ROOT", "cat",), ("sat",), (("cat", "the"),))
    test_step("RIGHT-ARC", "RA", ["ROOT", "run", "fast"], [], [],
              ("ROOT", "run",), (), (("run", "fast"),))


def test_parse():
    """Simple tests for the PartialParse.parse function
    Warning: these are not exhaustive
    """
    sentence = ["parse", "this", "sentence"]
    dependencies = PartialParse(sentence).parse(
        ["S", "S", "S", "LA", "RA", "RA"])
    dependencies = tuple(sorted(dependencies))
    expected = (('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this'))
    assert dependencies == expected,  \
        "parse test resulted in dependencies {:}, expected {:}".format(
            dependencies, expected)
    assert tuple(sentence) == ("parse", "this", "sentence"), \
        "parse test failed: the input sentence should not be modified"
    print("parse test passed!")


class DummyModel(object):
    """Dummy model for testing the minibatch_parse function
    First shifts everything onto the stack and then does exclusively right arcs if the first word of
    the sentence is "right", "left" if otherwise.
    """

    def predict(self, partial_parses):
        return [("RA" if pp.stack[1] is "right" else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]


def test_dependencies(name, deps, ex_deps):
    """Tests the provided dependencies match the expected dependencies"""
    deps = tuple(sorted(deps))
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(
            name, deps, ex_deps)


def test_minibatch_parse():
    """Simple tests for the minibatch_parse function
    Warning: these are not exhaustive
    """
    sentences = [["right", "arcs", "only"],
                 ["right", "arcs", "only", "again"],
                 ["left", "arcs", "only"],
                 ["left", "arcs", "only", "again"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[1],
                      (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[2],
                      (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
    test_dependencies("minibatch_parse", deps[3],
                      (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))
    print("minibatch_parse test passed!")


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        raise Exception(
            "You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
    elif args[1] == "part_c":
        test_parse_step()
        test_parse()
    elif args[1] == "part_d":
        test_minibatch_parse()
    else:
        raise Exception(
            "You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")