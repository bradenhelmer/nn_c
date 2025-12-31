# --- Compiler Settings ---
CC          := clang
NVCC        := nvcc
STD         := -std=c99
# Base flags used in all builds (Dependencies included via -MMD -MP)
CFLAGS      := $(STD) -Wall -Wextra -MMD -MP -D_POSIX_C_SOURCE=199309L

# --- CUDA Settings ---
CUDA_PATH   := /usr/local/cuda-13.0
CUDA_INC    := -I$(CUDA_PATH)/include
CUDA_LIB    := -L$(CUDA_PATH)/lib64 -lcudart -lcublas
NVCC_FLAGS  := -x cu --std=c99 --compiler-options -Wall

# --- Build-Specific Flags ---
# Debug: No optimization, debug symbols, debug macros
DEBUG_FLAGS := -g -O0 -DDEBUG
# Release: High optim, native arch, LTO, no debug macros
OPT_FLAGS   := -O3 -march=native -mavx512f -flto -DNDEBUG
# Flags for perf profiling
PERF_FLAGS  := -O3 -march=native -mavx512f -flto -g -fno-omit-frame-pointer -fno-inline -DNDEBUG -DPROFILING=1
# Flags for instruction level profiling
PROF_FLAGS  := -O3 -march=native -DPROFILING=1 -fprofile-instr-generate=mnist.profraw -fcoverage-mapping

LDFLAGS     := -lm $(CUDA_LIB)

# --- Directory Structure ---
SRC_DIR     := src
TEST_DIR    := tests
BUILD_DIR   := build

# Output Bins
BIN_DIR     := $(BUILD_DIR)/bin
OBJ_DIR     := $(BUILD_DIR)/obj
DEBUG_DIR   := $(BUILD_DIR)/debug_obj
PERF_DIR    := $(BUILD_DIR)/perf_obj
PROF_DIR    := $(BUILD_DIR)/prof_obj
TEST_OBJ_DIR:= $(BUILD_DIR)/test_obj
TEST_DBG_DIR:= $(BUILD_DIR)/test_debug_obj
GPU_OBJ_DIR := $(BUILD_DIR)/gpu_obj
GPU_DBG_DIR := $(BUILD_DIR)/gpu_debug_obj

# --- Source Discovery (Recursive) ---
# Uses 'find' to locate all .c files in SRC_DIR
SRCS        := $(shell find $(SRC_DIR) -name '*.c')
# Map source files to object files for each build type
OBJS        := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
DEBUG_OBJS  := $(patsubst $(SRC_DIR)/%.c, $(DEBUG_DIR)/%.o, $(SRCS))
PERF_OBJS   := $(patsubst $(SRC_DIR)/%.c, $(PERF_DIR)/%.o, $(SRCS))
PROF_OBJS   := $(patsubst $(SRC_DIR)/%.c, $(PROF_DIR)/%.o, $(SRCS))

# --- CUDA Source Discovery ---
CUDA_SRCS   := $(shell find $(SRC_DIR) -name '*.cu')
GPU_C_OBJS  := $(patsubst $(SRC_DIR)/%.c, $(GPU_OBJ_DIR)/%.o, $(SRCS))
GPU_CU_OBJS := $(patsubst $(SRC_DIR)/%.cu, $(GPU_OBJ_DIR)/%.cu.o, $(CUDA_SRCS))
GPU_OBJS    := $(GPU_C_OBJS) $(GPU_CU_OBJS)
GPU_DBG_C_OBJS  := $(patsubst $(SRC_DIR)/%.c, $(GPU_DBG_DIR)/%.o, $(SRCS))
GPU_DBG_CU_OBJS := $(patsubst $(SRC_DIR)/%.cu, $(GPU_DBG_DIR)/%.cu.o, $(CUDA_SRCS))
GPU_DBG_OBJS    := $(GPU_DBG_C_OBJS) $(GPU_DBG_CU_OBJS)

# --- Test Discovery ---
TEST_SRCS   := $(wildcard $(TEST_DIR)/*.c)
TEST_OBJS   := $(patsubst $(TEST_DIR)/%.c, $(TEST_OBJ_DIR)/%.o, $(TEST_SRCS))
TEST_DBG_OBJS := $(patsubst $(TEST_DIR)/%.c, $(TEST_DBG_DIR)/%.o, $(TEST_SRCS))

# --- Library Objects (Excluding main) ---
# Used for linking tests against the library code
LIB_OBJS       := $(filter-out $(OBJ_DIR)/main.o, $(OBJS))
LIB_DEBUG_OBJS := $(filter-out $(DEBUG_DIR)/main.o, $(DEBUG_OBJS))

# --- Targets ---
TARGET          := $(BIN_DIR)/neural_net
DEBUG_TARGET    := $(BIN_DIR)/neural_net_debug
PERF_TARGET     := $(BIN_DIR)/neural_net_perf
PROF_TARGET     := $(BIN_DIR)/neural_net_prof
TEST_TARGET     := $(BIN_DIR)/test_runner
TEST_DBG_TARGET := $(BIN_DIR)/test_runner_debug
GPU_TARGET      := $(BIN_DIR)/neural_net_gpu
GPU_DBG_TARGET  := $(BIN_DIR)/neural_net_gpu_debug

# --- Colors ---
GREEN  := \033[1;32m
BLUE   := \033[1;34m
YELLOW := \033[1;33m
RESET  := \033[0m

# ==============================================================================
#   RULES
# ==============================================================================

.PHONY: all clean run run-debug profile memcheck format help perf-build perf-record perf-report gpu gpu-debug run-gpu run-gpu-debug

all: $(TARGET) $(TEST_TARGET)

# --- Standard Optimized Build ---
$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	@echo "$(BLUE)Linking $@...$(RESET)"
	@$(CC) $(OPT_FLAGS) $(OBJS) -o $@ $(LDFLAGS)
	@echo "$(GREEN)Standard optimized build complete: $@$(RESET)"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) $(OPT_FLAGS) -I$(SRC_DIR) $(CUDA_INC) -c $< -o $@

# --- Debug Build ---
debug: $(DEBUG_TARGET)

$(DEBUG_TARGET): $(DEBUG_OBJS)
	@mkdir -p $(dir $@)
	@echo "$(BLUE)Linking debug build $@...$(RESET)"
	@$(CC) $(DEBUG_OBJS) -o $@ $(LDFLAGS)
	@echo "$(GREEN)Debug build complete: $@$(RESET)"

$(DEBUG_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@echo "Compiling (debug) $<..."
	@$(CC) $(CFLAGS) $(DEBUG_FLAGS) -I$(SRC_DIR) $(CUDA_INC) -c $< -o $@

# --- Perf Profiling Build ---
perf-build: $(PERF_TARGET)

$(PERF_TARGET): $(PERF_OBJS)
	@mkdir -p $(dir $@)
	@echo "$(BLUE)Linking perf build $@...$(RESET)"
	@$(CC) $(PERF_FLAGS) $(PERF_OBJS) -o $@ $(LDFLAGS)
	@echo "$(GREEN)Perf build complete: $@$(RESET)"

$(PERF_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@echo "Compiling (perf) $<..."
	@$(CC) $(CFLAGS) $(PERF_FLAGS) -I$(SRC_DIR) $(CUDA_INC) -c $< -o $@

# --- Instruction-level Profiling Build ---
profile: $(PROF_TARGET)

$(PROF_TARGET): $(PROF_OBJS)
	@mkdir -p $(dir $@)
	@echo "$(BLUE)Linking profile build $@...$(RESET)"
	@$(CC) -fprofile-instr-generate -flto $(PROF_OBJS) -o $@ $(LDFLAGS)
	@echo "$(GREEN)Profile build complete: $@$(RESET)"

$(PROF_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@echo "Compiling (profile) $<..."
	@$(CC) $(CFLAGS) $(PROF_FLAGS) -I$(SRC_DIR) $(CUDA_INC) -c $< -o $@

# --- GPU Build (CUDA) ---
gpu: $(GPU_TARGET)

$(GPU_TARGET): $(GPU_OBJS)
	@mkdir -p $(dir $@)
	@echo "$(BLUE)Linking GPU build $@...$(RESET)"
	@$(CC) $(OPT_FLAGS) $(GPU_OBJS) -o $@ $(LDFLAGS)
	@echo "$(GREEN)GPU build complete: $@$(RESET)"

# Compile C files with clang for GPU build (optimized)
$(GPU_OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@echo "Compiling (GPU) $<..."
	@$(CC) $(CFLAGS) $(OPT_FLAGS) -I$(SRC_DIR) $(CUDA_INC) -c $< -o $@

# Compile CUDA files with nvcc (optimized)
$(GPU_OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	@echo "Compiling (CUDA) $<..."
	@$(NVCC) $(NVCC_FLAGS) -O3 -I$(SRC_DIR) $(CUDA_INC) -c $< -o $@

# --- GPU Debug Build ---
gpu-debug: $(GPU_DBG_TARGET)

$(GPU_DBG_TARGET): $(GPU_DBG_OBJS)
	@mkdir -p $(dir $@)
	@echo "$(BLUE)Linking GPU debug build $@...$(RESET)"
	@$(CC) $(DEBUG_FLAGS) $(GPU_DBG_OBJS) -o $@ $(LDFLAGS)
	@echo "$(GREEN)GPU debug build complete: $@$(RESET)"

# Compile C files with clang for GPU debug build
$(GPU_DBG_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@echo "Compiling (GPU debug) $<..."
	@$(CC) $(CFLAGS) $(DEBUG_FLAGS) -I$(SRC_DIR) $(CUDA_INC) -c $< -o $@

# Compile CUDA files with nvcc (debug)
$(GPU_DBG_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	@echo "Compiling (CUDA debug) $<..."
	@$(NVCC) $(NVCC_FLAGS) -g -G -O0 -I$(SRC_DIR) $(CUDA_INC) -c $< -o $@

# --- Tests ---
test: $(TEST_TARGET)
	@echo "$(YELLOW)Running tests...$(RESET)"
	@./$(TEST_TARGET)

$(TEST_TARGET): $(LIB_OBJS) $(TEST_OBJS)
	@mkdir -p $(dir $@)
	@$(CC) $(LIB_OBJS) $(TEST_OBJS) -o $@ $(LDFLAGS)

$(TEST_OBJ_DIR)/%.o: $(TEST_DIR)/%.c
	@mkdir -p $(dir $@)
	@$(CC) $(CFLAGS) -I$(SRC_DIR) $(CUDA_INC) -c $< -o $@

test-debug: $(TEST_DBG_TARGET)
	@echo "$(YELLOW)Running debug tests...$(RESET)"
	@./$(TEST_DBG_TARGET)

$(TEST_DBG_TARGET): $(LIB_DEBUG_OBJS) $(TEST_DBG_OBJS)
	@mkdir -p $(dir $@)
	@$(CC) $(LIB_DEBUG_OBJS) $(TEST_DBG_OBJS) -o $@ $(LDFLAGS)

$(TEST_DBG_DIR)/%.o: $(TEST_DIR)/%.c
	@mkdir -p $(dir $@)
	@$(CC) $(CFLAGS) $(DEBUG_FLAGS) -I$(SRC_DIR) $(CUDA_INC) -c $< -o $@

# --- Utilities ---

run: all
	@./$(TARGET)

run-debug: debug
	@./$(DEBUG_TARGET)

run-gpu: gpu
	@./$(GPU_TARGET)

run-gpu-debug: gpu-debug
	@./$(GPU_DBG_TARGET)

run-profile: profile
	@./$(PROF_TARGET)
	llvm-profdata merge -output=mnist.profdata mnist.profraw
	llvm-profdata show --topn=20 mnist.profdata

lldb: debug
	@echo "Launching lldb..."
	@lldb $(DEBUG_TARGET)

lldb-test: $(TEST_DBG_TARGET)
	@echo "Launching lldb for tests..."
	@lldb $(TEST_DBG_TARGET)

memcheck: debug
	valgrind --exit-on-first-error=yes --leak-check=full --show-leak-kinds=all --track-origins=yes $(DEBUG_TARGET)

memcheck-test: $(TEST_DBG_TARGET)
	valgrind --leak-check=full --show-leak-kinds=all $(TEST_DBG_TARGET)

format:
	@echo "$(BLUE)Formatting source files...$(RESET)"
	@find $(SRC_DIR) $(TEST_DIR) -name '*.c' -o -name '*.h' | xargs clang-format -i
	@echo "$(GREEN)Formatting complete$(RESET)"

clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(RESET)"
	@rm -rf $(BUILD_DIR) perf.data perf.data.old

help:
	@echo "Available targets:"
	@echo "  all          : Build default target"
	@echo "  debug        : Build with debug symbols"
	@echo "  gpu          : Build GPU-accelerated version with CUDA"
	@echo "  gpu-debug    : Build GPU version with debug symbols"
	@echo "  test         : Build and run tests"
	@echo "  clean        : Remove build artifacts"
	@echo "  format       : Run clang-format"

# --- Perf profiling Workflow ---
perf-record: perf-build
	@echo "$(YELLOW)Recording perf profile...$(RESET)"
	@sudo perf record -g -F 4000 --call-graph dwarf ./$(PERF_TARGET)
	@echo "$(GREEN)Profile saved to perf.data$(RESET)"

perf-report:
	@echo "$(YELLOW)Generating perf report...$(RESET)"
	@sudo perf report --stdio --no-children -n | head -100


# Include dependency files
-include $(OBJS:.o=.d)
-include $(DEBUG_OBJS:.o=.d)
-include $(PERF_OBJS:.o=.d)
-include $(PROF_OBJS:.o=.d)
-include $(TEST_OBJS:.o=.d)
-include $(TEST_DBG_OBJS:.o=.d)
-include $(GPU_C_OBJS:.o=.d)
-include $(GPU_DBG_C_OBJS:.o=.d)
