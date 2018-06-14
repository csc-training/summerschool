#include <stdio.h>

int main()
{
    FILE *fp;
    char str[5];

    /* opening file for reading */
    fp = fopen("file.txt", "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: file.txt did not exist\n");

        return (-1);
    }
    fgets(str, 5, fp);
    printf("%s", str);
    //fputs is a simpler function that just writes
    //a string to the file
    //fputs(str, stdout);

    fclose(fp);

    return (0);
}
