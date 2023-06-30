# Parallel I/O
### Striping example:
Different tools to setting and displaying stripe properties
 - `lfs setstripe` Set striping properties of a directory
or new file
 - `lfs getstripe`  Return information on current
striping settings
 - `lfs df`  Show disk usage of this file system


```
touch first_file
lfs getstripe first_file

mkdir experiments
lfs setstripe -c 4 experiments
touch experiments/new_file
lfs getstripe experiments/new_file
```

