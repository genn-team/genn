import nbformat as nbf
import re

# Comile regular expressions used to find GeNN installation code
_gdown_re = re.compile(r"!gdown ([_0-9a-zA-Z]+)")
_pip_re = re.compile(r"!pip install pygenn-([0-9]+.[0-9]+.[0-9]+)-cp([0-9]+)-cp([0-9]+)-linux_x86_64.whl")

def process_notebooks(notebooks, gdown_hash, pygenn_ver, python_tag, callback=None):
    # Loop through notebooks 
    for notebook_path in argv[1:-3]:
        # Open notebook
        print(f"Processing notebook {notebook_path}")
        notebook = nbf.read(notebook_path, nbf.NO_CONVERT)

        # Loop through cells
        for i, c in enumerate(notebook.cells):
            # Skip non-code cells
            if c["cell_type"] != "code":
                continue
            
            # Search for gdown and pip install lines
            source = c["source"]
            gdown_match = _gdown_re.search(source)
            pip_match = _pip_re.search(source)
            if gdown_match and pip_match:
                print(f"\tGeNN installed in cell {i}")
                print(f"\tWheel hash {gdown_match[1]}")
                print(f"\tOld GeNN version {pip_match[1]}")
                print(f"\tOld Python version {pip_match[2]}")
                
                assert pip_match[2] == pip_match[3]
                
                # Update gdown and pip lines with new version
                source = _gdown_re.sub(f"!gdown {gdown_hash}", source)
                source = _pip_re.sub(f"!pip install pygenn-{pygenn_ver}-cp{python_tag}-cp{python_tag}-linux_x86_64.whl", source)
                
                if callback:
                    source = callback(source)

                # Replace source and stop searching
                c["source"] = source
                break

        nbf.write(notebook, notebook_path, version=nbf.NO_CONVERT)

if __name__ == "__main__":
    from sys import argv

    assert len(argv) > 4

    # Extract fixed arguments from end of list after potential wildcard expansion
    gdown_hash = argv[-3]
    pygenn_ver = argv[-2]
    python_tag = argv[-1]
    
    notebooks = argv[1:-3]
    
    process_notebooks(notebooks, gdown_hash, pygenn_ver, python_tag)
