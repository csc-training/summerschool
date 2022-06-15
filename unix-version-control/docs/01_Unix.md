---
title:  Introduction to Unix
event:  CSC Summer School in High-Performance Computing 2022
lang:   en
---


# Unix skills needed for the summer school

- To be able to follow the summer school, please make sure that you:
    - know how to start and use a ***command line interface*** (that is,
      console, shell, terminal)
    - know what is meant by ***directories*** and ***files*** and understand
      ***unix file paths***
    - know what ***environment variables*** are and how to use them
    - know how to ***change directories*** (cd),
      ***list the contents of a directory*** (ls),
      ***move and copy files*** (cp, mv), and ***remove files*** (rm)
    - know how to ***use a text editor*** (nano, vim, emacs or similar)


# Unix philosophy: Do one thing and do it well

- All input and output of different tools should be single flat files,
  so that they can be combined through the use of pipes
- Examples:
    - count how many filenames contain the string `model` in the current
      directory and all it's subdirectories:
      ```bash
      find . | grep model | wc -l
      ```
    - concatenate multiple files while replacing all occurrences of the
      string `/some/path` with `/another/path`:
      ```bash
      cat *.sh | sed -e "s#/some/path#/another/path#g" > output.sh
      ```


# Skills needed to work with supercomputers

- Logging in to, and copying files from/to a supercomputer (ssh, scp)
- Setting up an environment to compile and run programs
    - Module system (before compiling)
    - Batch queue system (for running)
- Compiling codes
    - Basic use of version control (git)
    - Make
    - Compilers (gcc)


# SSH: log in and copy files to/from remote hosts

- The `ssh` command is used to log into a remote host
    - You will be asked for your password to establish a connection
  ```bash
  ssh user@host
  ```
    - Additionaly, ssh can also be used to run a command on the host:<br>
      `ssh user@host command`

- The `scp` command is used to copy files and directories between different
  hosts on a network
  ```bash
  scp user@host:source target   # remote->local
  scp source user@host:target   # local->remote
  ```


# Hands-on: ssh and scp

- Create a text file on your computer (e.g. `name.txt`) with your first
  name as the content
- Copy the file to Puhti using `scp`:
  ```bash
  scp local.txt trainingxxx@puhti.csc.fi:~/
  ```
- Log in to Puhti using `ssh`, check that the file is there, and modify it
  (for example add your last name)
- Copy the modified file back to your local computer and check that the file
  has been changed
