#!/usr/bin/env python

"""
Pandoc filter to add link to MPI reference to MPI functions
"""
import functools
import re
import yaml

from pathlib import Path
from pandocfilters import toJSONFilter, Code, Link, Str


with Path(__file__).with_name("valid_mpi_links.yaml").open() as fd:
    MPI_LINKS = yaml.safe_load(fd.read())
    MPI_LINKS["functions"] = set(MPI_LINKS["functions"])


def link_mpi_element(mpi_func, element):
    """Add MPI link element around pandoc element.

    Parameters
    ----------
    mpi_func
        Name of MPI function to be linked
    element
        Pandoc element encapsulated by the link

    Returns
    -------
    Pandoc element
    """
    url = f'https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/{mpi_func}.3.html'
    return Link(("", [], [["target", "_blank"], ["style", "color:var(--csc-grey)"]]),
                [element],
                [url, ""])


def link_mpi_text(text, Element, MPIElement):
    """Add links to MPI functions detected in the text.

    Parameters
    ----------
    text
        Text to be processed
    Element
        Pandoc constructor for non-MPI elements
    MPIElement
        Pandoc constructor for MPI elements

    Returns
    -------
    List of pandoc elements
    """

    # We search for MPI functions with this regexp, but
    # not all matches are valid functions, which makes
    # the code somewhat complicated
    prog = re.compile(r"MPI_\w+")
    text_done = ""  # text before the current re match
    text_todo = text  # text after the current re match
    elements = []
    while True:
        m = prog.search(text_todo)
        if m is None:
            text_done += text_todo
            if text_done != "":
                elements.append(Element(text_done))
            break
        func = m.group(0)
        part, text_todo = text_todo.split(func, 1)
        text_done += part
        if func in MPI_LINKS["functions"]:
            if text_done != "":
                elements.append(Element(text_done))
                text_done = ""
            elements.append(link_mpi_element(func, MPIElement(func)))
        else:
            text_done += func

    return elements


def add_mpi_links(key, value, format, meta):
    """Pandoc filter adding links to to MPI functions."""

    if key == 'Str':
        return link_mpi_text(value,
                             Str,
                             functools.partial(Code, ("", ["mpi"], [])),
                             )

    if key == 'Code':
        ((ident, classes, kvs), contents) = value
        if "mpi" in classes:
            # Code elements with mpi class are already processed.
            # We get infinite recursion without this check.
            return
        return link_mpi_text(contents,
                             functools.partial(Code, (ident, classes, kvs)),
                             functools.partial(Code, (ident, classes + ["mpi"], kvs)),
                             )


if __name__ == "__main__":
    toJSONFilter(add_mpi_links)
