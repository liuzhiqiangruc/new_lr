CC = gcc -std=c99 -O3

all: lr deeplr

lr : lr.o auc.o data.o hash.o regcfg.o regr.o str.o main_lr.o
	$(CC) $^ -lm -o $@

deeplr : deeplr.o auc.o data.o hash.o regcfg.o regr.o str.o main_deeplr.o
	$(CC) $^ -lm -o $@


main_lr.o : main.c
	cc -c main.c -DLR -o main_lr.o

main_deeplr.o : main.c
	cc -c main.c -DDEEPLR -o main_deeplr.o


clean:
	rm *.o
	rm lr
	rm deeplr
