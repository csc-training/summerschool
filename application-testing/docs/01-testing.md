---
title:  Testing software
event:  CSC Summer School in High-Performance Computing 2024
lang:   en
---

# Overview
- Motivation
- Types of tests
- Demo
- Summary

# Motivation {.section}

# Why test your software?
Why use

::: incremental
- version control?
- linting?
- code formatting tools?
- out-of-source build systems?
- Vim instead of emacs?
:::

# Why test your software?
Because it makes (among other things)

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
:::::: {.column width="60%"}
::: incremental
A test is not a unit test if ^[Michael Feathers https://www.artima.com/weblogs/viewpost.jsp?thread=126923]:

- It talks to the database
- It communicates across the network
- It touches the file system
- It can't run at the same time as any of your other unit tests
- You have to do special things to your environment (such as editing config files) to run it
:::
::::::
:::::: {.column width="40%"}
<small>
```cpp
struct Field {
    Field(
        std::vector<double> &&temperatures,
        int num_rows,
        int num_cols);

    double sum() const;
    // ...
};

TEST(field_test, zero_field_sum_is_zero) {
    constexpr int num_rows = 2000;
    constexpr int num_cols = 2000;
    const Field field(
        std::vector<double>(num_rows * num_cols),
        num_rows,
        num_cols
    );

    ASSERT_EQ(field.sum(), 0.0);
}
```
</small>
::::::
:::::::::

# Unit tests
\
\
\
Unit tests run **fast** so you can run them **often**

# Integration tests
::::::::: {.columns}
:::::: {.column width="60%"}
::: incremental
- Test integrated parts of the program for correctness
- Can vary in size
    - operation of a single module
    - co-operation of multiple modules
    - operation of an entire library or application
- May look like unit tests, but may interact with 'external' code
- May take multiple seconds or minutes to complete
:::
::::::
:::::: {.column width="40%"}
<small>
```cpp
std::tuple<int, int, std::vector<double>>
  read_field(const std::string &filename);

TEST(io_test, read_field_data_from_file) {
    auto [num_rows, num_cols, data] =
        read_field("testdata/bottle.dat");
    ASSERT_EQ(data.size(), 40000);
    ASSERT_EQ(num_rows * num_cols, data.size());
}
```
</small>
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
Ensure the program doesn't regress in

::: incremental
- performance
    - *it's slower than before*
    - benchmarks
- usability
    - *it doesn't compile anymore*
    - tests of public API
- correctness
    - *it suddenly started crashing*
    - unit & integration tests
:::

# Demo {.section}

# Summary
- Motivation
    - Testing makes your life easier and improves your program's quality
- Types of tests
    - Unit tests
    - Integration tests
    - Regression tests
- Demo
