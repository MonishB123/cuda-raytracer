CC = nvcc
NVCC = nvcc
# -rdc=true might be needed if we have device code linking, but for now simple structure is fine.
# We need to explicitly include vcpkg include path.
COMMON_FLAGS = -O3 -arch=sm_60 -I./vcpkg/installed/x64-windows/include -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH

# Libraries
# Link against jsoncpp.lib in vcpkg
LIBS = -L./vcpkg/installed/x64-windows/lib -ljsoncpp

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = .

# Files
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/common/*.cpp)
CU_SRCS = $(wildcard $(SRC_DIR)/cuda/*.cu)

CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
CU_OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SRCS))

# Targets
TARGET = $(BIN_DIR)/main

all: $(TARGET)

$(TARGET): $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) $(COMMON_FLAGS) -o $@ $^ $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(COMMON_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(COMMON_FLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET).exe

.PHONY: all clean