
<!--
SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Exercise: Errors in kernel

if you notice, some idx happen twice. that is because we used the wrong stride (HEIGHT instead of WIDTH) when evaluating the idx variable to assign the value.
In this case it was easy, but sometimes knowing what each thread is doing can be very useful to debug... 
