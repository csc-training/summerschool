# Simple git usage

## Basic workflow
1. `git pull` - update changes from the remote repository
2. **Modify** and **create** files as you wish
3. `git add file1` add files into the commit
4. `git commit -m "commit message"` issue the commit
5. `git push` upload changes back to the remote repository

In addition `git status` can be invoked any time to check what is happening with the repository.

Some other useful commands and shortcuts:
- `git rm file1` can be used to **remove** files from the directory and version control
- `git reset file1` can be used to **unstage** file from a commit (in case you accidentally added a wrong file)
- `git  checkout -- file1` discards changes made for `file1`
- `git add -u` adds all files modified and currently under version control
- `git log` shows the commit history

## Keeping your repository up-to-date
Because the `summerschool` repository is a fork now (i.e., it was not created by you, but you *forked* it) you can keep it up-to-date by issuing
```
git pull https://github.com/csc-training/summerschool.git
```
In case you modified same files as the teachers happened to update, you (usually) want to specify that all the changes made by **you** should be kept. In that case, use
```
git pull https://github.com/csc-training/summerschool.git -X ours
```
Alternatively, if you want the git to prefer the new upcoming files in case of an merge conflict, you can use `-X theirs`.

## Configuring git
Should be run only once, when logging in to a new computer:
```
git config --global user.name "github_username"
git config --global user.email email_used@in_github_account.com
```

## Cloning the remote repository
Cloning, i.e., downloading a remote repository to your local computer:
```
git clone https://github.com/csc-training/summerschool.git
```

## Stashing
Sometimes it is useful to not discard but **stash** your local changes. Difference here is that you can re-apply the modifications later on.
```
git stash
```
And to re-apply them
```
git stash apply
```
If you want to apply one of your older stashes, you can specify it by naming it like this: `git stash apply stash@{2}`. List of all the stashes can be shown with `git stash list`.









