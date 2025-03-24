# deanonymization-types
The source code folder will contain 3 separate folders, each containing the graphs and the Python program used to perform a specific type of deanonymization.
## Setup & Execution
- First, download and access the source code folder using the code editor Visual Studio Code.
- Next, create a virtual environment to test the code:
  - Start by clicking the search bar above the code, and select “Show and Run Commands”. Then, click “Python: Select Interpreter”. For environment type, select “Venv”.
  - Next, select the interpreter path, “\.venv\Scripts\python.exe”
  - If “requirements.txt” shows up, be sure to check that.
  - It will install everything in requirements.txt into the .venv
- After the venv is finished installing, type “.venv/Scripts/Activate.ps1” into the terminal to activate the virtual machine. 
- Change directories to whichever folder(pertaining to a de-anonymization technique) you want to test first (seed-based, seed-free, etc). 
- Then, run the python file in the folder that corresponds to the type of deanonymization (Ex: “python seed-based.py” produces a complete mapping of nodes between G1 & G2 with seed-based deanonymization) 
- Also, while in the virtual environment, type “deactivate” in the terminal to exit the virtual environment.

