## Multiple files ##

1) Implement the functions of fourth part of the exercise
[../datastructures-functions-heat](../datastructures-functions-heat)
in a separate source file and write also a header file providing the
function declarations. Include the header file and call the routines
in the main program (You may start also from the model solution of
exercise
[../datastructures-functions-heat/solution](../datastructures-functions-heat/solution))

2) The files [pngwriter.h](pngwriter.h) and [pngwriter.c](pngwriter.c)
provide a utility function `save_png` for writing out a
two-dimensional array in the `.png` format. Investigate the header
file [pngwriter.h](pngwriter.h) and use the utility function for
writing out the array of fourth part of exercise
[../datastructures-functions-heat/](../datastructures-functions-heat/). You
can also start from the skeleton code in [libpng.c](libpng.c). You
need to include the header file, compile also the file
[pngwriter.c](pngwriter.c) and link your program against the `libpng`
library, that is, use linker option `-lpng`. Try to build the
executable first directly in the command lind and then with the
provided `Makefile` (use command `make`). The resulting png-file can
be viewed using for example `eog` program.
