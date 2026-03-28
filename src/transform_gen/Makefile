# Compiler and flags
CXX = clang++
CXXFLAGS = -std=c++17 -fno-exceptions -fno-rtti
INCLUDES = -I$(MLIR_INCLUDE_DIR)
LDFLAGS = -L$(MLIR_LIB_DIR) -lMLIR -lMLIRTransform -lMLIRLinalg -lMLIRStandardOps

# Paths to MLIR installation (update these to match your environment)
MLIR_INCLUDE_DIR = /path/to/mlir/include
MLIR_LIB_DIR = /path/to/mlir/lib

# Source and target
SRC = tile_transform.cpp
TARGET = tile_transform

# Default target
all: $(TARGET)

# Compile the program
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LDFLAGS)

# Clean up
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean