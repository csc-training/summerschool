---
title:  Introduction to C
author: CSC Summerschool 
date:   2019-07
lang:   en
---

# Control structures and loops {.section}

# Control structures // if – else 

<div class="column">
```c
if (i < 100) {
    data[i] = x[i] + c;
} else {
    data[i] = x[i] - c;
}
```


- if (_condition_) {`TRUE`{.input}} else {`FALSE`{.output}}
- negation:  **!**
- non-zero value == TRUE
</div>

<div class="column">
- Test operators:
  : == equal to
  : != not equal to
  : && AND
  : || OR
  : \< less than
  : \> greater than
  : \<= less or equal
  : \>= greater or equal
</div>



# Control structures // if – else
<div class=column>

```c
// simple if-else
if (x > 1.2) {
    y = 4*x*r; // TRUE
} else {
    y = 0.0;   // FALSE
}


// else is optional
if (x || y) {
    z += x + y;
}
```
</div>

<div class=column>

```c
// complex if-elseif-else

if ( (x > 1.2) && (i != 0) ){
	y = x / i;     // 1st TRUE
} else if ( x < 1.0 ) {
	y = -x;        // 1st FALSE
	               // 2nd TRUE
} else {
	y = 0.0;       // 1st, 2nd FALSE
}




```
</div>



# Control structures // switch
```c
switch (...) {        // conditional expression 
    case (...):       // test value
        ...           // do something
        break;        // end of branch
    default:          // default branch executed if nothing else matches
        ...           // do something
}
```
- condition:
    - single variable OR complex expression

- branch (_= case_) with matching value chosen

- break stops branch



# Control structures // switch

```c
 // simple switch based on the
 // value of integer i
 switch (i) {
     case 1:
         printf("one\n");
         break;
     case 2:
         printf("two\n");
         break;
     default:
         printf("many\n");
         break;              // good style to break even the last branch
    }

```



# Control structures // switch

<div class=column>
```c
switch (i) {								A
    case 1:
        printf("one\n");
    case 2:
        printf("two\n");
    default:
        printf("many\n");
}
```
```c
switch (i) {								B
    case 1:
        printf("one\n");
        break;
    case 2:
        printf("two\n");
        break;
    default:
        printf("many\n");
}
```

</div>

<div class=column>

```c
switch (i) {								C
    default:
        printf("many\\n");
    case 1:
        printf("one\\n");
        break;
    case 2:
        printf("two\\n");
        break;
}
```

In each case, what would be printed on the screen if i is 1, 2, or 3?

</div>



# Control structures // ? : operator

```c
Exp1 ? Exp2 : Exp3;
```

The value of a ? expression is determined like this:

- Exp1 is evaluated.
    - If it is `TRUE`{.input}, then Exp2
    - If Exp1 is `FALSE`{.output}, then Exp3



# Loops // while
- **while (_condition_) {_code block_}**

```c
i = 0;
// loop i = 0..99 with an increment of one
while ( i < 100 ) {
    data[i] = x[i] + c;
    i++;
}
```
- Code block executes repeatedly as long as condition is TRUE

- Condition executed _before_ iteration

- **do {_code block_} while (_condition_)**

    - Block executed at least once



# Loops // for
- **for (\<init\>; \<condition\>; \<after\>) {...}**

```c
// loop i = 0..99 with an increment of one
for ( i = 0; i < 100; i++ ) {
    data[i] = x[i] + c;
}
```
* **\<init\>**        execute once before entering the loop
* **\<condition\>**   stop loop when condition is FALSE
* **\<after\>**       execute after each iteration


# Loops // for

```c
 // loop using a for-statement
 // i is incremented after each iteration
 for (i=0; i<bar; i++) {
     x += i*3.14 + c;
     c = i + 2;
 }
```
 
```c
 // the same loop but with a while-statement
 i=0;
 while (i < bar) {
     x += i*3.14 + c;
     c = i + 2;
     i++;
 }
```


# Jump statements

<div class=column>
- **break**

    - end a loop (for, while, do) or a switch statement

- **continue**

    - continue with the next iteration of a loop

</div>
<div class=column>

```c
// jump out of a loop
for (i=0; i\<10; i++) {
    printf("in loop");
    if (i == x)
        break;
}
```

```c
// jump to the next iteration
for (i=0; i\<10; i++) {
    if (i \< x)
        continue;
    printf("in loop");
}
```

</div>



# Jump statements

<div class=column>
- **goto**

    - jump to a labeled statement in the same function
</div>
<div class=column>


```c
// jump to a labeled statement
int a = 10;
LOOP:do
{
    if( a == 15) {
        a = a + 1;
        goto LOOP;
    }
    a++;
}while( a < 20 );
```
</div>



# Summary

- Conditional branching: **if-else**, **switch**

- Looping as long as condition is true: **while**, **do – while**

- Fixed number of loop passes: **for**

- Jump statements: **continue**, **break**, **goto**

