# Parallel I/O
### Striping example:

```
touch first_file
lfs getstripe first_file

mkdir stripe_exp
lfs setstripe -c 4 experiments
touch experiments/new_file
lfs getstripe experiments/new_file
```
