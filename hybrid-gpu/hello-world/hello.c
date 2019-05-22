#include <stdio.h>
#ifdef _OPENACC
#include <openacc.h>
#endif

int main(void) {
#ifdef _OPENACC
    acc_device_t devtype;
#endif

    printf("Hello world!\n");
#ifdef _OPENACC
    devtype = acc_get_device_type();
    printf("Number of available OpenACC devices: %d\n",
            acc_get_num_devices(devtype));
    printf("Type of available OpenACC devices: %d\n", devtype);
#else
    printf("Code compiled without OpenACC\n");
#endif

    return 0;
}
