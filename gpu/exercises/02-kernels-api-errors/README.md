
## Exercise: Errors from API calls

In the previous exercise we learned to query values from the API: `const auto result = hipDeviceGetAttribute(&value, attribute, device);`.
The API gives us the value for the attribute we query through a reference: `&value`. Why doesn't it just return it?
Because it returns something else, and we chose to call it `result`.

Many of the API calls actually return an error code of type `hipError_t`


We already encountered this earlier when we discussed catching errors
from kernel launches in a [previous exercise](../02-kernels-kernel-launch-wrapper).

It's very useful to catch errors from the API calls.
This can save you *a lot* of debugging time later on.

So how to do it? It's very similar to the way we did it at kernel launch:
```cpp
auto result = hipDeviceGetAttribute(&value, attribute, device);
if (result != hipSuccess) {
    printf("Error in %s at line %d\n%s: %s\n",
           __FILE__,
           __LINE__,
           hipGetErrorName(result),
           hipGetErrorString(result));
    exit(EXIT_FAILURE);
}
```

This is one way to do it, but it can get tedious when you call the API multiple times.
Also, it makes the source code very annoying to read, when the important lines are hidden
in the error checking clutter:
```cpp
result = hipDeviceGetAttribute(&value, attribute1, device);
if (result != hipSuccess) {
    printf("Error in %s at line %d\n%s: %s\n",
           __FILE__,
           __LINE__,
           hipGetErrorName(result),
           hipGetErrorString(result));
    exit(EXIT_FAILURE);
}

result = hipDeviceGetAttribute(&value, attribute2, device);
if (result != hipSuccess) {
    printf("Error in %s at line %d\n%s: %s\n",
           __FILE__,
           __LINE__,
           hipGetErrorName(result),
           hipGetErrorString(result));
    exit(EXIT_FAILURE);
}

result = hipDeviceGetAttribute(&value, attribute3, device);
if (result != hipSuccess) {
    printf("Error in %s at line %d\n%s: %s\n",
           __FILE__,
           __LINE__,
           hipGetErrorName(result),
           hipGetErrorString(result));
    exit(EXIT_FAILURE);
}
```

Due to these reasons, it's recommended and very common to handle the error checking with a function,
and, like with kernel calls, wrap the API call with a macro:
```cpp
HIP_ERRCHK(hipDeviceGetAttribute(&value, attribute1, device));
HIP_ERRCHK(hipDeviceGetAttribute(&value, attribute2, device));
HIP_ERRCHK(hipDeviceGetAttribute(&value, attribute3, device));
```

with
```cpp
#define HIP_ERRCHK(result) hip_errchk(result, __FILE__, __LINE__)

static inline void hip_errchk(hipError_t result, const char *file,
                              int32_t line) {
    if (result != hipSuccess) {
        printf("Error in %s at line %d\n%s: %s\n",
            file,
            line,
            hipGetErrorName(result),
            hipGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}
```

This way the error checking is out of the way, but it's still there.

In this exercise there are a bunch of problems with API calls.
Wrap the API calls with the error checking macro and fix any errors you get during runtime!


HINT:
Instead of doing this
```cpp
auto result = hipApiCall(args);
```

do this
```cpp
HIP_ERRCHK(hipApiCall(args));
```
