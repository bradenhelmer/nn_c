# Compiler and flags
CC = clang
CFLAGS = -Wall -Wextra -O2 -std=c99
DEBUGFLAGS = -g -O0 -DDEBUG
LDFLAGS = -lm  # Link math library

# Directories
SRC_DIR = src
BUILD_DIR = build
TEST_DIR = tests
BIN_DIR = $(BUILD_DIR)/bin
OBJ_DIR = $(BUILD_DIR)/obj
DEBUG_OBJ_DIR = $(BUILD_DIR)/debug_obj
TEST_OBJ_DIR = $(BUILD_DIR)/test_obj
TEST_DEBUG_OBJ_DIR = $(BUILD_DIR)/test_debug_obj

# Source files
SRCS = $(wildcard $(SRC_DIR)/**/*.c) $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
DEBUG_OBJS = $(patsubst $(SRC_DIR)/%.c, $(DEBUG_OBJ_DIR)/%.o, $(SRCS))

# Test files
TEST_SRCS = $(wildcard $(TEST_DIR)/*.c)
TEST_OBJS = $(patsubst $(TEST_DIR)/%.c, $(TEST_OBJ_DIR)/%.o, $(TEST_SRCS))
TEST_DEBUG_OBJS = $(patsubst $(TEST_DIR)/%.c, $(TEST_DEBUG_OBJ_DIR)/%.o, $(TEST_SRCS))

# Exclude main.c when building tests
LIB_OBJS = $(filter-out $(OBJ_DIR)/main.o, $(OBJS))
LIB_DEBUG_OBJS = $(filter-out $(DEBUG_OBJ_DIR)/main.o, $(DEBUG_OBJS))

# Targets
TARGET = $(BIN_DIR)/neural_net
DEBUG_TARGET = $(BIN_DIR)/neural_net_debug
TEST_TARGET = $(BIN_DIR)/test_runner
TEST_DEBUG_TARGET = $(BIN_DIR)/test_runner_debug

# Default target
all: directories $(TARGET) $(TEST_TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(OBJ_DIR)/activations
	@mkdir -p $(OBJ_DIR)/data
	@mkdir -p $(OBJ_DIR)/linalg
	@mkdir -p $(OBJ_DIR)/nn
	@mkdir -p $(OBJ_DIR)/training
	@mkdir -p $(OBJ_DIR)/utils
	@mkdir -p $(DEBUG_OBJ_DIR)
	@mkdir -p $(DEBUG_OBJ_DIR)/activations
	@mkdir -p $(DEBUG_OBJ_DIR)/data
	@mkdir -p $(DEBUG_OBJ_DIR)/linalg
	@mkdir -p $(DEBUG_OBJ_DIR)/nn
	@mkdir -p $(DEBUG_OBJ_DIR)/training
	@mkdir -p $(DEBUG_OBJ_DIR)/utils
	@mkdir -p $(TEST_OBJ_DIR)
	@mkdir -p $(TEST_DEBUG_OBJ_DIR)

# Build main executable
$(TARGET): $(OBJS)
	@echo "Linking $@..."
	@$(CC) $(OBJS) -o $@ $(LDFLAGS)
	@echo "Build complete: $@"

# Build object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

# Build debug executable
$(DEBUG_TARGET): $(DEBUG_OBJS)
	@echo "Linking debug build $@..."
	@$(CC) $(DEBUG_OBJS) -o $@ $(LDFLAGS)
	@echo "Debug build complete: $@"

# Build debug object files
$(DEBUG_OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling (debug) $<..."
	@$(CC) $(CFLAGS) $(DEBUGFLAGS) -c $< -o $@

# Test target
test: directories $(TEST_TARGET)
	@echo "Running tests..."
	@./$(TEST_TARGET)

$(TEST_TARGET): $(LIB_OBJS) $(TEST_OBJS)
	@echo "Building tests..."
	@$(CC) $(LIB_OBJS) $(TEST_OBJS) -o $@ $(LDFLAGS)

$(TEST_OBJ_DIR)/%.o: $(TEST_DIR)/%.c
	@echo "Compiling test $<..."
	@$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

# Debug test target
test-debug: directories $(TEST_DEBUG_TARGET)
	@echo "Running debug tests..."
	@./$(TEST_DEBUG_TARGET)

$(TEST_DEBUG_TARGET): $(LIB_DEBUG_OBJS) $(TEST_DEBUG_OBJS)
	@echo "Building debug tests..."
	@$(CC) $(LIB_DEBUG_OBJS) $(TEST_DEBUG_OBJS) -o $@ $(LDFLAGS)

$(TEST_DEBUG_OBJ_DIR)/%.o: $(TEST_DIR)/%.c
	@echo "Compiling test (debug) $<..."
	@$(CC) $(CFLAGS) $(DEBUGFLAGS) -I$(SRC_DIR) -c $< -o $@

# Debug build
debug: directories $(DEBUG_TARGET)

# Run the program
run: all
	@./$(TARGET)

# Run debug build
run-debug: debug
	@./$(DEBUG_TARGET)
# Debug with lldb
lldb: debug
	@echo "Launching lldb..."
	@lldb $(DEBUG_TARGET)

# Debug tests with lldb
lldb-test: test-debug
	@echo "Launching lldb for tests..."
	@lldb $(TEST_DEBUG_TARGET)

# Memory check with valgrind (if installed)
memcheck: debug
	valgrind --leak-check=full --show-leak-kinds=all $(DEBUG_TARGET)

memcheck-test: $(TEST_DEBUG_TARGET)
	valgrind --leak-check=full --show-leak-kinds=all $(TEST_DEBUG_TARGET)

# Format code with clang-format
format:
	@echo "Formatting source files..."
	@find $(SRC_DIR) $(TEST_DIR) -name '*.c' -o -name '*.h' | xargs clang-format -i
	@echo "Formatting complete"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)

# Print variables (useful for debugging the Makefile)
print-%:
	@echo $* = $($*)

# Phony targets
.PHONY: all clean test test-debug debug run run-debug lldb lldb-test memcheck directories format

# Dependencies (auto-generate)
DEPFLAGS = -MMD -MP
CFLAGS += $(DEPFLAGS)
DEPS = $(OBJS:.o=.d)
-include $(DEPS)
