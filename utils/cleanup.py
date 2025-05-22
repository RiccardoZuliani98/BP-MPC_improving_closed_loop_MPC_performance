import os, glob

def cleanup():
    """
    This function cleans the casadi jit files in the main directory
    """
    
    # get all jit files
    list_of_files = glob.glob("./jit*")

    # get all tmp files
    list_of_files.extend(glob.glob("./tmp*"))

    # eliminate all
    for filename in list_of_files:
        os.remove(filename)