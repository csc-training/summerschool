#include <cstdio>
#include <vector>

struct Foo {
    bool condition;
    int value;
    double x;
    std::vector<float> collection;

    template <typename T>
    void some_func(std::vector<T> &);
    void do_something(int);
};

void do_many_things1(Foo foo, Foo bar) {
    // Do first thing
    int a = 123;
    while (foo.condition) {
        a += 1;
        bar.do_something(a);
        // .
        // .
        // .
    }

    // Do second thing
    for (auto i : foo.collection) {
        if (i > bar.x) {
            // .
            // .
            // .
        } else {
            // .
            // .
            // .
        }
    }

    // Do third thing
    foo.some_func(bar.collection);
    // .
    // .
    // .
    // .
    // .

    // Output things
    printf("foo: %i \n", foo.value);
    printf("bar: %i \n", bar.value);
}

void do_first_thing(int, Foo, Foo);
void do_second_thing(Foo, Foo);
void do_third_thing(Foo, Foo);
void output_things(Foo);

void do_many_things(Foo foo, Foo bar) {
    do_first_thing(123, foo, bar);
    do_second_thing(foo, bar);
    do_third_thing(foo, bar);
    output_things(foo);
    output_things(bar);
}
