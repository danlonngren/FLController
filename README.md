# FLController
Simple Fuzzy logic controller in c++ which uses no virual inheritence or complex dynamic memory allocations other than the rules are stored as a vector. Which makes it suitable for small embedded platforms.



## Run commands
```bash
# Clean build
rm -rf build

# Generate build
cmake -DBUILD_TESTS=ON -S . -B build

# Build project
cmake --build build

# Run googletests
GTEST_COLOR=1 ctest --test-dir build --output-on-failure --j 12

# Single command
rm -rf build && cmake -DBUILD_TESTS=ON -S . -B build && cmake --build build && \
GTEST_COLOR=1 ctest --test-dir build --output-on-failure --j 12