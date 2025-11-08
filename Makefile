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
TEST_OBJ_DIR = $(BUILD_DIR)/test_obj

# Source files
SRCS = $(wildcard $(SRC_DIR)/**/*.c) $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

# Test files
TEST_SRCS = $(wildcard $(TEST_DIR)/*.c)
TEST_OBJS = $(patsubst $(TEST_DIR)/%.c, $(TEST_OBJ_DIR)/%.o, $(TEST_SRCS))

# Exclude main.c when building tests
LIB_OBJS = $(filter-out $(OBJ_DIR)/main.o, $(OBJS))

# Targets
TARGET = $(BIN_DIR)/neural_net
TEST_TARGET = $(BIN_DIR)/test_runner

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(OBJ_DIR)/linalg
	@mkdir -p $(OBJ_DIR)/activations
	@mkdir -p $(OBJ_DIR)/utils
	@mkdir -p $(TEST_OBJ_DIR)

# Build main executable
$(TARGET): $(OBJS)
	@echo "Linking $@..."
	@$(CC) $(OBJS) -o $@ $(LDFLAGS)
	@echo "Build complete: $@"

# Build object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

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

# Debug build
debug: CFLAGS += $(DEBUGFLAGS)
debug: clean all

# Run the program
run: all
	@./$(TARGET)

# Memory check with valgrind (if installed)
memcheck: debug
	valgrind --leak-check=full --show-leak-kinds=all ./$(TARGET)

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
.PHONY: all clean test debug run memcheck directories format

# Dependencies (auto-generate)
DEPFLAGS = -MMD -MP
CFLAGS += $(DEPFLAGS)
DEPS = $(OBJS:.o=.d)
-include $(DEPS)
