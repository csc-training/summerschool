Wrap the API calls with the error checking macro and fix any errors you get during runtime!

Instead of doing this
```cpp
auto result = hipApiCall(args);
```

do this
```cpp
HIP_ERRCHK(hipApiCall(args));
```
