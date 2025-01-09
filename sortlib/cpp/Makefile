CXX = g++
CXXFLAGS = -std=c++17 -Wall -Iinclude

SRC = src/test_sorting.cpp
INCLUDE = include/sorting.h include/sorting.tpp
BUILD = build
TARGET = $(BUILD)/test_sorting

$(TARGET): $(SRC) $(INCLUDE)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD)

.PHONY: run clean