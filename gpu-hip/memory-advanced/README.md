Extra exercise: Advanced C++ example of memory wrapping: the exercise with this code is to try to understand what is going on.
Lots of comments are in the code. The basic idea is to wrap all the hip primitives into cpp templated classes, in order to forget about mallocs/frees and follow the C++ RAII (resource allocation is initialization) principle. advanced c++ features (such as fold expressions and tuple conversions) are used to avoid to introduce a lot of copy/paste errors and keep the overall structure as flexible as possible.

you can compile this example on lumi with: 

CC -xhip --std=c++17 main.cpp
