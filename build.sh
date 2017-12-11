cc -c *.c -O3 -std=c99
cc *.o -o lr -lm
mv lr data
rm *.o
