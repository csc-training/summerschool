# Documenting and testing {.section}

# Why to document your code?

- You will forget details
    - Code that you wrote 6 months ago is often indistinguishable from code 
      that someone else has written.
- Writing documentation may improve the design of your code
- Have other people to use (and cite!) your code
    - if the documentation is not good enough, people will not use it    
- Allow other people to contribute to development
    - practically impossible without documentation

# What to document?

- How to use the code
    - installation instructions
    - format of the input file and possible values of input parameters
    - tutorials on specific cases
    - examples that can be copy-pasted
    - FAQs
    - How to cite the code !

# What to document?

- How to develop the code
    - equations implemented by the code when appropriate
        - mapping of physical symbols to variables in code
    - coding style
    - instructions for contributing
    - APIs (application programming interfaces)
    - Implementation details

# How to document?

- Documentation is part of the code
- Keep the documentation close to source
    - Same versioning as for the code
    - Guides in e.g. `docs` subdirectory
    - APIs in the source code, e.g. describe arguments of subroutine next to
      the defition
        - many tools can generate automatically API documention from comments
    - Non-obvious implementation choices in comments in source code
- Tools for documentation: RST and Markdown markup languages, wikis, Doxygen

# How is your code documented?

**Discuss** within your table!

# Documentation in GPAW & PlasmaBox

<div class=column>
GPAW
- User developer guides, tutorials etc. written in RST
- www-site generated with Sphinx
- APIs in Python docstrings
- <https://wiki.fysik.dtu.dk/gpaw>
</div>

<div class=column>
PlasmaBox
- TODO
</div>

# Testing
