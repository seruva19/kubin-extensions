import os


def rename_checkpoint(old_name, new_name):
    try:
        os.rename(old_name, new_name)
    except FileNotFoundError:
        print(f"error: checkpoint {old_name} not found.")
    except Exception as e:
        print(f"error renaming file: {e}")
