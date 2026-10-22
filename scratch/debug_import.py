import sys
import traceback

try:
    import main
    print("Main imported without error!")
except Exception as e:
    with open("scratch/import_error.txt", "w") as f:
        traceback.print_exc(file=f)
    print("Caught exception:", e)
