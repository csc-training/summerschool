# Input/Output {.section}

# Outline

- Input/output (I/O) formatting
- File I/O
    - File opening and closing
    - Writing and reading to/from a file
    - Formatted and unformatted (binary) files
    - Stream I/O
- Internal I/O

# Input/Output formatting

- To prettify output and to make it human readable, use format
  descriptors in connection with the write statement
- Although less often used nowadays, it can also be used with read to
  input data at fixed line positions and using predefined field
  lengths
- Used through format statements, character variable or embedded in
  read/write fmt keyword

# Output formatting

<small>
<center>

| Data type          | Format descriptors | Examples                |
| --                 | --                 | --                      |
| Integer            | iw, iw.m           | `write(*,'(i5)') j`     |
|                    |                    | `write(*,'(i5.3)') j`   |
|                    |                    | `write(*,'(i0)') j`     |
| Real (decimal and  | fw.d               | `write(*,'(f7.4)') r`   |
| exponential forms, | ew.d               | `write(*,'(e12.3)') r`  |
| auto-scaling)      | gw.d               | `write(*,'(g20.13)') r` |
| Character          | a, aw              | `write(*,'(a)') c`      |
| Logical            | lw                 | `write(*,'(l2)') l`     |

</center>

w=width of the output, d=number of digits after the decimal point, 
m=minimum number of characters  

</small>

# Output formatting: miscellaneous

- With complex numbers provide format for both real and imaginary
  parts:
```fortran
complex :: z 
write (*,'(f6.3,2x,f6.3)') z ! real and imaginary parts
```
- Line break and whitespace:
``` fortran
write(*,'(f6.3,/,f6.3)') x, y ! linebreak between x and y
write(*,'(i3,2x,f6.3)') i, x  ! 2 spaces between i and x
```
- It is possible that an edit descriptor will be repeated a specified
    number of times
``` fortran
write(*,'(5i8)') ivec(1:5)
write(*,'(4(i5,2x,f8.3))') (ivec(j),zvec(j),j=1,4)
```

# The I0 and G0 format descriptors

- Dynamic sizing of **real** and **integer** valued output
    - **I0** appeared in F03 and G0 was introduced in F08

- Output fields are left justified with all the unnecessary leading blanks
  (and precision for **real** valued variables) removed
``` fortran
integer :: i = 12345
real (kind=4) :: sp = 1.23e0
real (kind=8) :: dp = 1.234567890d0
write(*,fmt='("<i=",i0,", reals=",g0,1x,g0,">")') i,sp,dp
```
- Output is **`<i=12345, reals=1.230000 1.234567890000000>`**

# Handling character strings

- Fortran provides several intrinsic functions for handling character
  strings, such as

    - **`trim(string)`** - removes blank spaces from the end of string
    - **`adjustl(string)/adjustr(string)`** - moves blank spaces from the
      beginning/end of the string to the end/beginning of it
    - **`len(string)`** - length of a string
    - **`index(string, substring)`** - returns the starting position of a
      substring within a string

# Opening and closing files: basic concepts

- By now, we have written to terminal or read from keyboard with <br>
  `write(*, ...)` and `read(*, ...)`
- Writing to or reading from a file is similar with few differences:
    - File must be opened with an OPEN statement, in which the unit number
      and (optionally) the file name are given
    - Subsequent writes (or reads) must refer to the given unit number
    - File should be closed at the end

# Opening and closing a file

- The syntax is (the brackets [ ] indicate optional keywords or
  arguments)
```
open([unit=]iu, file='name' [, options])
close([unit=]iu [, options])
```
- For example
``` fortran
open(10, file= 'output.dat', status='new')
close(unit=10, status='keep')
```

# Opening and closing a file

- The first parameter is the unit number
- The keyword unit= can be omitted
- The unit numbers 0, 5 and 6 are predefined
    - 0 is output for standard (system) error messages
    - 5 is for standard (user) input
    - 6 is for standard (user) output
    - These units are opened by default and should not be re-opened nor
      closed by the user
- If the file name is omitted in the OPEN, the file name will based on
  unit number, e.g. ’fort.12’ for `unit=12`

# File opening options

- **status**: existence of a file
    - 'old', 'new', 'replace', 'scratch', 'unknown'
- **position**: offset, where to start writing
    - 'append'
- **action**: file operation mode
    - 'write', 'read', 'readwrite'
- **form**: text or binary file
    - 'formatted', 'unformatted'

# File opening options

- **access**: direct or sequential file access
    - 'direct', 'sequential', 'stream',
- **iostat**: error indicator, (output) integer
    - non-zero only upon an error
- **recl**: record length, (input) integer
    - for direct access files only
    - warning (check): may be in bytes *or* words

# File opening: file properties

- Use the **inquire** statement to find out information about
    - file existence
    - file unit open status
    - various file attributes
- The syntax has two forms, one based on file name, the other for unit
  number
    -  `inquire(file='name', options ...)` or
    -  `inquire(unit=iu, options ...)`

# File opening: file properties

- **exist** does file exist? (logical)

- **opened** is file / unit opened? (logical)

- **form** 'formatted' or 'unformatted' (char)

- **access** 'sequential' or 'direct' or 'stream' (char)

- **action** 'read', 'write', 'readwrite' (char)

- **recl** record length (integer)

- **size** file size in bytes (integer)


# File opening: file properties example

``` fortran
! check the existence of a file
logical :: file_exist
inquire(file='foo.dat', exist=file_exist)
if (.not. file_exist) then
  write(*,*) 'the file does not exist'
else
  ! do something with the file foo.dat
endif
```

# File writing and reading

- Writing to and reading from a file is done by giving the corresponding
  unit number `(iu)` as a parameter :
``` fortran
write(iu,*) str
write(unit=iu,fmt=*) str
read(iu,*) str
read(unit=iu, fmt=*) str
```

- Formats and other options can be used as needed
- If keyword **`unit`** is used, also the keyword **`fmt`** must be used
    - Note: **`fmt`** is applicable to formatted, text files only

- The star format (`*`) indicates list-directed output
  (i.e. programmer does not choose the input/output styles)

# Formatted vs. unformatted files

- Text or *formatted* files are
    - Human readable
    - Portable i.e. machine independent
- Binary or *unformatted* files are
    - Machine readable only, generally not portable
    - Much *faster to access* than formatted files
    - Suitable for large amount of data due to *reduced file sizes*
    - Internal data representation used for numbers, thus no number
      conversion, no rounding of errors compared to formatted data

# Unformatted I/O

- Write to a sequential binary file
``` fortran
real :: rval
character(len=60) :: string
open(10, file='foo.dat', form='unformatted')
write(10) rval
write(10) string
close(10)
```
- No format descriptors allowed
- Reading similarly
``` fortran
read(10) rval
read(10) string
```

# Stream I/O

- A binary file write adds extra record delimiters (hidden from
  programmer) to the beginning and end of records
- In Fortran 2003 using access method **'stream'** avoids this and
  implements a C-like approach
- It is recommended to use stream I/O
- Create a stream (binary) file
``` fortran
real :: dbheader(20), dbdata(300)
open(10,file='my_database.dat', access='stream')
write(10) dbheader
```
- Reading similarly

# Internal I/O

- Often it is necessary to filter out data from a given character string
    - Or to pack values into a character string
- Fortran internal I/O with `read` and `write` becomes now handy: instead of
  unit number one provides character array as first argument
     - Actual files are not involved at all

```fortran
character(len=13) :: cl1
character(len=60) :: cl2
integer :: njobs, istep

! extract a number from character string 
cl1 = 'time step\# 10'
read(cl1,fmt='(10x,i3)') istep
! write data to a character string
njobs = 2014
write(cl2,'(a,i0)') 'the number of jobs completed = ', njobs
```

# Command line input

- In many cases, it is convenient to give parameters for the program
  directly during program launch
    - Instead of using a parser, reading from an input file etc.
- Fortran 2003 provides a way for this
    - **`command_argument_count()`**
    - **`get_command_argument(integer i, character arg(i))`**


# Command line input

<div class="column">
- Example: reading in two integer values from the command line
- The (full) program should be launched as (e.g.):
```shell
$ ./a.out 100 100
```
</div>
<div class="column">
``` fortran
subroutine read_command_line(height, width)
  integer, intent(out) :: height, width
  character(len=10) :: args(2)
  integer :: n_args, i
  n_args = command_argument_count()
  if ( n_args /= 2 ) then
    write(*,*) ' Usage : ./exe height width'
    call abort()
  end if
  do i = 1, 2
    call get_command_argument(i,args(i))
    args(i) = trim(adjustl(args(i)))
  end do
  read(args(1),*) height
  read(args(2),*) width
end subroutine read_command_line
```
</div>

# Summary

- Input/Output formatting
- Files: communication between a program and the outside world
    - Opening and closing a file
    - Data reading & writing
- Use unformatted (stream) I/O for all except text files
