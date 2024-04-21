C_SOURCES = $(wildcard matrix/*.c neural/*.c util/*.c *.c)
HEADERS = $(wildcard matrix/*.h neural/*.h util/*.h *.h)
OBJ = ${C_SOURCES:.c=.o}
CFLAGS = -fopenmp

MAIN = main
CC = gcc
# CC = /usr/bin/gcc
LINKER = /usr/bin/ld

# run: ${MAIN}
# 	./${MAIN}

main: ${OBJ}
	${CC} ${LDFLAGS} ${CFLAGS} $^ -o $@ -lm

# Generic rules
%.o: %.c ${HEADERS}
	${CC} ${CFLAGS} -c $< -o $@ -lm

clean:
	rm matrix/*.o *.o neural/*.o util/*.o ${MAIN}
	rm results/${MAIN}/*
	rm -r testing_net
