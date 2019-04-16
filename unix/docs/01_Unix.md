---
title:  Introduction to Unix
author: CSC Summerschool 
date:   2019-07
lang:   en
---


# Basic use of Unix-like systems
- To be able to follow the upcoming content, please make sure that you have the relevant basic Unix skills. In particular, check that
	- you know how to start and/or use a command line interface (that is, console, shell, terminal)
	- you know how and when to use at least these basic commands: ls, cp, mv, rm
	- you can use some text editor, such as nano, vim, emacs

# "Do one thing and do it well"
- This is the core principle of the Unix philosophy
- In accordance to it, all input and output of the different tools should be single flat files...
- ... so that they can be combined through the use of pipes
- Example:  
	`$ ls -l | grep key | less`


# Basic skills for working with supercomputers
- Logging in to, and copying files from and to the supercomputer, i.e. a remote host: ssh and scp
- Basic use of version control
- Make
- Setting up the environment to compile and run programs
	- Module system (before compiling)
	- Batch queue system (for running)

# Logging in and moving files
- The ssh command is used for logging into a remote host 
- You will be asked for your password to establish a connection
- Additionaly, ssh can be used to run a command on the host  
`$ ssh [user@]host`
- The scp command is used for secured copying of files and folders across different hosts on a network
- You  will be asked for your password here as well  
`$ scp [user@]host:source target # remote->local`  
`$ scp source [user@]host:target # local->remote`

# Hands-on: ssh and scp 

- Create a local text file, for example, "name.txt" with your first name as the contents
- Copy your text file to Sisu using scp  
	`$ scp local.txt trng99@sisu.csc.fi:~/`
- Login to Sisu using ssh, check that file is there, and modify it: for example, add your last name to it
- Copy the modified file back to your local computer, and check that the file has changed.
