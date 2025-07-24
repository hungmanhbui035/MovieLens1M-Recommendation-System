import pandas as pd

# Method 1: Current method (what you're using)
print("=== Current pandas method ===")
movies_df = pd.read_csv("./ml-1m/movies.dat", sep="::", header=None, 
                       names=["movie_id", "title", "genres"], 
                       engine="python", encoding="latin-1")
print(f"Rows in dataframe: {len(movies_df)}")

# Method 2: Count actual lines in file
print("\n=== Counting file lines ===")
with open("./ml-1m/movies.dat", 'r', encoding='latin-1') as f:
    lines = f.readlines()
    print(f"Total lines in file: {len(lines)}")
    
# Method 3: Manual parsing to see what fails
print("\n=== Manual parsing test ===")
failed_lines = []
parsed_count = 0

with open("./ml-1m/movies.dat", 'r', encoding='latin-1') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        parts = line.split("::")
        if len(parts) != 3:
            failed_lines.append((i+1, line, len(parts)))
        else:
            parsed_count += 1

print(f"Successfully parsed: {parsed_count}")
print(f"Failed to parse: {len(failed_lines)}")

if failed_lines:
    print("\nFirst few problematic lines:")
    for line_num, content, part_count in failed_lines[:5]:
        print(f"Line {line_num}: {part_count} parts - {content[:100]}...")

# Method 4: Try with different pandas settings
print("\n=== Alternative pandas parsing ===")
try:
    movies_df2 = pd.read_csv("./ml-1m/movies.dat", sep="::", header=None, 
                            names=["movie_id", "title", "genres"], 
                            engine="python", encoding="latin-1",
                            quoting=3)  # QUOTE_NONE
    print(f"With quoting=3: {len(movies_df2)} rows")
except Exception as e:
    print(f"quoting=3 failed: {e}")

try:
    movies_df3 = pd.read_csv("./ml-1m/movies.dat", sep="::", header=None, 
                            names=["movie_id", "title", "genres"], 
                            engine="c", encoding="latin-1")
    print(f"With C engine: {len(movies_df3)} rows")
except Exception as e:
    print(f"C engine failed: {e}") 