CC = mpicc
CFLAGS = 
TARGETS = main_int main_float
NUM_PROCESSES = 3

all: $(TARGETS)

main_int: main_int.c
	$(CC) $(CFLAGS) -o main_int main_int.c

main_float: main_float.c
	$(CC) $(CFLAGS) -o main_float main_float.c

run_int: main_int
	mpirun -np $(NUM_PROCESSES) ./main_int

run_float: main_float
	mpirun -np $(NUM_PROCESSES) ./main_float

run_int_print: main_int
	mpirun -np $(NUM_PROCESSES) ./main_int --print

run_float_print: main_float
	mpirun -np $(NUM_PROCESSES) ./main_float --print

run_int_no_seq: main_int
	mpirun -np $(NUM_PROCESSES) ./main_int --no-seq

run_float_no_seq: main_float
	mpirun -np $(NUM_PROCESSES) ./main_float --no-seq

run_int_print_no_seq: main_int
	mpirun -np $(NUM_PROCESSES) ./main_int --print --no-seq

run_float_print_no_seq: main_float
	mpirun -np $(NUM_PROCESSES) ./main_float --print --no-seq

run_all: all run_int run_float

clean:
	rm -f $(TARGETS)

.PHONY: all run_int run_float run_int_print run_float_print run_int_no_seq run_float_no_seq run_int_print_no_seq run_float_print_no_seq run_all clean
