import subprocess

# Run pytest tests in the /Users/cinnamon/Downloads/Project03_46W38/tests/ directory
test_command = ['pytest', '/Users/cinnamon/Downloads/Project03_46W38/tests/test_main_script.py']
process = subprocess.run(test_command, capture_output=True, text=True)

print(process.stdout)
if process.stderr:
    print(process.stderr)

if process.returncode == 0:
    print("All tests passed successfully.")
else:
    print("Some tests failed. Check the output above for details.")