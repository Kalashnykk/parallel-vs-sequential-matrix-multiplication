CC = mpicc
CFLAGS = 
TARGETS = main_int main_float
NP = 3

all: $(TARGETS)

main_int: main_int.c
	$(CC) $(CFLAGS) -o main_int main_int.c

main_float: main_float.c
	$(CC) $(CFLAGS) -o main_float main_float.c

# compute flags based on make variables
PRINT_FLAG := $(if $(PRINT),--print)
NOSEQ_FLAG := $(if $(NOSEQ),--no-seq)
NOPAR_FLAG := $(if $(NOPAR),--no-par)
FLAGS := $(PRINT_FLAG) $(NOSEQ_FLAG) $(NOPAR_FLAG)

run_int: main_int
	mpirun -np $(NP) ./main_int $(FLAGS)

run_float: main_float
	mpirun -np $(NP) ./main_float $(FLAGS)

run_all: all run_int run_float

clean:
	rm -f $(TARGETS)

.PHONY: all run_int run_float run_all clean
