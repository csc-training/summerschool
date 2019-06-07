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

- Documentation should be part of the code
- Keep the documentation close to source
    - Same versioning as for the code
    - Guides in e.g. `docs` subdirectory
    - APIs in the source code, e.g. describe arguments of subroutine next to
      the defition
        - many tools can generate automatically API documention from comments
    - Non-obvious implementation choices in comments in source code
- Tools for documentation: RST and Markdown markup languages, wikis, Doxygen, 
  github pages and readthedocs for hosting

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

**Simulations and analysis with untested software <br>
do not constitute science!**

<p>

- Experimental scientist would never conduct an experiment with uncalibrated 
  detectors
- Computational scientist should never conduct simulations with untested 
  software


# Why software needs to be tested?

- Ensure expected functionality
- Ensure expected functionality is preserved
    - software is fragile, bugs lurk in easily
    - in large projects innocent looking changes can have surprising side 
      effects
- Testing helps detecting errors early
- Testing is essential for reproducibility of results
- Tests make is easier to verify whether software is correctly installed

# Defensive programming

- Would you trust a software ...
    - ... when its tests do not pass?
    - ... if the tests are never run?
    - ... if there are no tests at all?
- Assume mistakes will happen and introduce guards agains them
- Test drive development
    - Tests are written before actually implementing the functionality

# What should be tested in software ?

- Validity of physical model
    - comparison to experimental results
- Validity of numerical methods and implementation
    - Comparison to analytic solutions of special cases
    - Conservation laws and symmetries
- Correct functioning of individual subroutines and modules
- Performance
    - changes in software may lead into degragation in performance

# Unit testing

- Tests for smallest testable part of an application
    - function, module or class
- Ideally, tests are independent of each other
- Frameworks: cUnit, cppunit, pFUnit, Google Test, pytest, ...
- Client code which executes tests can be also hand-written
- Unit testing helps in managing complexity
    - well structured and modular code is easy to test

# Integration testing

- Combining individual software modules and testing as group
    - "trivial" changes in code can have surprising side effects
    - example: testing a specific solver which utilizes several modules
- At the highest level tests the whole application
- Recommended to be combined with unit testing

# Challenges with HPC

- Behavior can depend on the number of threads and MPI tasks
    - Parallel components should be tested with multiple different 
      parallelization schemes
- Large scale integration tests can be time consuming
- Changes in program code may also lead to degradation in performance and 
  scalability
     - tests should track also the performance
- Performance is often system/architecture specific
    - preferably test on multiple architectures

# Continuous integration

- Automatic testing
    - build test ensures that code is always in a "working state"
- Unit and integration tests can be run automatically after build succeeds
- Nightly / weekly tests
    - Software is built and tested at specified times
- Test at "commit"
    - test set is run when new code is pushed to main repository
- Nightly and "commit" tests can exclude time consuming tests

# Continuous integration

<div class=column>
- Test system should send a notification when test fails
    - mailing list, IRC, Flowdock, …
- Test status can be shown also in www-page
- Tools for continuous integration:
    - TravisCI: widely used, nice integration to Github
    - Jenkins: support for complex automation tasks
    - GitlabCI
</div>
<div class=column>
![](images/ci-flowchart.svg){.center width=90%}
</div>

# How is your code tested?

**Discuss** within your table!

# Testing in GPAW & PlasmaBox

<div class=column>
GPAW

- Wide test set
    - Unit tests and integration tests
- Continuous integration with GitlabCI
- Developers should run tests manually before pushing
- Tests set should run after installation

</div>

<div class=column>
PlasmaBox

- TODO

</div>

# Take home messages

- Document your code
- Test your code, prefer automatic testing


