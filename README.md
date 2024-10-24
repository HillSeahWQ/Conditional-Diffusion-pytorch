### Setup Prerequisites

1. **Install Python 3.12.3**
2. **Install Poetry**
3. **Install Nvidia CUDA 12.1**
   - Note: The version of PyTorch in this project uses CUDA 12.1 for GPU computing.

---

### Steps to Run

1. **Install Dependencies:**

   ```bash
   poetry install
   ```
2. **Enter the virtual environment:**

   ```bash
   poetry shell
   ```
3. **Run any file**

---

### Troubleshooting

1. **Issue:** When running `import torch`, you encounter the error `[WinError 126] The specified module could not be found. Error loading "~\torch\lib\fbgemm.dll" or one of its dependencies.`

   **Solution:** 

   - Download and install the **Visual Studio C/C++ IDE and Compiler** for Windows (Community Edition) from [Microsoft's official website](https://visualstudio.microsoft.com/vs/features/cplusplus/).
   - For detailed guidance, watch [YouTube tutorial](https://www.youtube.com/watch?v=sbQPGyVbePY&t=185s)
