---
title:  Testing software
event:  CSC Summer School in High-Performance Computing 2024
lang:   en
---

# Overview
- Why?
- Different types of tests
- Testing frameworks
- Demo
- Summary

# Why? {.section}

# Why test your software?
::: incremental
- Why use version control?
- Why use linting?
- Why use code formatting tools?
- Why use out-of-source build systems?
- Why use Vim and purge emacs from your OS?
:::

# Why test your software?
Because it makes

::: incremental
- you more productive
- your life easier
- creation of **well working** and **easy to maintain** software easier
:::

# Test types {.section}

# Test types
::: incremental
- Unit tests
- Integration tests
- Regression tests
- Others (security, usability, acceptance, performance...)
:::

# Unit tests
::::::::: {.columns}
:::::: {.column width="50%"}
::: incremental
A test is not a unit test if[^1]:

- It talks to the database
- It communicates across the network
- It touches the file system
- It can't run at the same time as any of your other unit tests
- You have to do special things to your environment (such as editing config files) to run it
:::
::::::
:::::: {.column width="50%"}
```cpp
Logger::Logger(const std::string &fname,
               bool add_location):
    fname(out_name),
    add_location(add_location)
{}

TEST(Logger_constructed_correctly) {
    // Arrange
    const auto fname = "log.log";
    const auto add_location = true;
    const Logger logger(fname, add_location);

    // [Act]

    // Assert
    ASSERT(logger.fname == fname);
    ASSERT(logger.add_location == add_location);
}
```
::::::
:::::::::

[^1]: Michael Feathers https://www.artima.com/weblogs/viewpost.jsp?thread=126923

# Unit tests
\
\
\
Unit tests run **fast** so you can run them **often**

# Integration tests
::::::::: {.columns}
:::::: {.column width="50%"}
::: incremental
- Test integrated parts of the program for correctness
- Can vary in size
    - operation of a single module
    - co-operation of multiple modules
    - operation of an entire library or application
:::
::::::
:::::: {.column width="50%"}
```cpp
Logger::write(const std::string &message);

TEST(Logger_writes_message) {
    // Arrange
    const auto fname = "log.log";
    const auto add_location = true;
    const Logger logger(fname, add_location);
    const auto message =
        "This is logged to file";

    // Act
    logger.write(message);

    // Assert
    const auto contents =
        std::read_to_string(fname);
    ASSERT(contents.contains(message) == true);
}
```
::::::
:::::::::

# Integration tests
\
\
**Don't** run after every change
\
\
**Do** run as part of CI pipeline after every commit or pull request


# Regression tests
- TODO

# Frameworks {.section}

# Frameworks
A long list from which to choose from

- https://en.wikipedia.org/wiki/List_of_unit_testing_frameworks

We'll be focusing of googletest (with CTest) today

# Googletest

- TODO

# Demo {.section}

# Summary
- Different types of tests
    - Unit tests
    - Integration tests
    - Regression tests
- Testing software and frameworks
    - Googletest
- Demo
