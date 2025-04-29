# Memory management strategies

The purpose of this exercise is to compare 6 different memory management
strategies and their computational overhead. The following functions are
called at the end of this file by the `main()` function:

* The function `explicitMem()` represents a basic explicit memory management strategy
* The function `explicitMemPinned()` represents an explicit memory management strategy with pinned host memory
* The function `explicitMemNoCopy()` represents an explicit memory management strategy where the data can reside at GPU memory during an iterative loop (no recurring memory copies needed)
* The function `unifiedMem()` represents a basic unified memory management strategy
* The function `unifiedMemPrefetch()` represents a unified memory management strategy with prefetching
* The function `unifiedMemNoCopy()` represents a unified memory management strategy where the data can reside at GPU memory during an iterative loop (no recurring memory copies needed)

The task is to fill the missing function calls in the code indicated by lines beginning with `#error`, and followed by a descriptive instruction. To compile this, rocm/5.3.3 or higher is needed, older rocm modules do not support memory asynchronous malloc/free (they hang)
